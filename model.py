from torchmetrics import AUROC, Accuracy, F1Score
from transformers import AutoModel, Adafactor
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np

import torch
import einops

from losses import NCEandRCE, NGCEandMAE

from modules import (FFLayer, FFResLayer, FuseHead, NormalizeProject, ReduceFuseHead, ReductionHead,
                     AttentionFeatureMatrixFuser, 
                     ForwardModuleList, 
                     LossBalancer, 
                     EMABalancer, 
                     AlignedSimilarityFuser, ReductionHead, 
                     SimilarityFuser,
                     WeightedHead,
                     TransformerExtraFuser,
                     FuseExtraHead,
                     FFHead
                     )
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import AugmentedWrapperDataset

def logit_norm(fun, x, y, t=1.):
    norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
    ln = torch.div(x, norms) / t
    return fun(ln, y)

def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False

def train_late_fusion(model, batch):
    encoded_batch, labels=batch

    multitask=len(labels.shape) > 1

    if not multitask:
        labels=labels.unsqueeze(-1)

    x_hat_list=model(**encoded_batch)

    losses=[]
    for x_hat, y, f in zip(x_hat_list,
                           labels.unbind(axis=-1),
                           model.loss_functions):
        loss=f.to(model.device)(x_hat, y)
        losses.append(loss)

    if model.loss_balancer is not None: 
        loss=model.loss_balancer(torch.stack(losses))
    else:
        loss=torch.nanmean(torch.stack(losses))
    model.log("train_loss", torch.nanmean(torch.stack(losses)))
    return loss

def train_hybrid_fusion(model, batch):
    encoded_batch, labels=batch

    multitask=len(labels.shape) > 1

    if not multitask:
        labels=labels.unsqueeze(-1)

    multi, image, text = model(**encoded_batch)

    losses=[]
    for m, i, t, y, f in zip(multi, image, text,
                             labels.unbind(axis=-1),
                             model.loss_functions):
        m_loss = f.to(model.device)(m, y)
        i_loss = f.to(model.device)(i, y)
        t_loss = f.to(model.device)(t, y)
        loss = m_loss + i_loss + t_loss# + f_loss
        losses.append(loss)

    if model.loss_balancer is not None: 
        loss=model.loss_balancer(torch.stack(losses))
    else:
        loss=torch.nanmean(torch.stack(losses))
    model.log("train_loss", torch.nanmean(torch.stack(losses)))
    return loss


class DefaultModel(L.LightningModule):
    def __init__(self, encoder, class_weights, training_steps,
                 dropout=.5,
                 lr=1e-4,
                 frozen=8,
                 semi_supervised=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.learning_rate = lr
        self.num_warmup_steps = int(0.1*training_steps)
        self.num_training_steps = training_steps
        self.frozen = frozen
        self.dropout = dropout
        self.encoder = encoder
        self.eps = 1e-8
        self.semi_supervised = semi_supervised
        self.loss_functions = [nn.CrossEntropyLoss(weight=torch.tensor(cw).float()) for cw in class_weights]
        self.output_list = [len(cw) for cw in class_weights]
        self.loss_balancer = LossBalancer(len(self.output_list))

        self.configure_transformer()

    def log_eval(self, preds, labels, split='valid'):
        losses=[]
        scores=[]

        multitask=len(labels.shape) > 1

        if not multitask:
            labels=labels.unsqueeze(-1)
        multi, image, text = preds
        for m, i, t, y, f, in zip(multi, image, text,
                                    labels.unbind(axis=-1),
                                     self.loss_functions):
            x_hat = torch.softmax(m,-1) + torch.softmax(i,-1) + torch.softmax(t,-1)
            loss = f.to(self.device)(x_hat, y)
            x_p = x_hat.argmax(-1)
            x_hat_alt = F.softmax(x_hat.detach(), dim=-1)
            x_hat_alt = x_hat_alt[:, -1] if x_hat.shape[-1] == 2 else x_hat_alt

            y_d = y.detach().cpu().numpy()
            idx = np.where(y_d != -100)
            y_d = y_d[idx]
            x_d = x_p.detach().cpu().numpy()[idx]
            x_h_d = x_hat_alt.cpu().float().numpy()[idx]

            score = (f1_score(y_d, x_d, average='macro',),
                     accuracy_score(y_d, x_d),
                     roc_auc_score(y_d, x_h_d,
                                   labels=np.arange(x_hat.shape[-1]),
                                   average='macro', multi_class='ovo',
                                   ),
                     )
            losses.append(loss)
            scores.append(score)

        loss = torch.nanmean(torch.stack(losses))
        self.log(f"{split}_loss", loss)
        for i, s in enumerate(scores):
            f1, acc, auc = s
            self.log(f"{split}_{i}_acc", acc)
            self.log(f"{split}_{i}_f1", f1)
            self.log(f"{split}_{i}_auc", auc)

        if len(scores) > 1:
            f1s, accs, aucs = list(zip(*scores))
            self.log(f"{split}_mean_acc", sum(accs)/len(accs))
            self.log(f"{split}_mean_f1", sum(f1s)/len(f1s))
            self.log(f"{split}_mean_auc", sum(aucs)/len(aucs))
        else:
            f1s, accs, aucs = scores[0]
            self.log(f"{split}_mean_acc", accs)
            self.log(f"{split}_mean_f1", f1s)
            self.log(f"{split}_mean_auc", aucs)

    def training_step(self, batch, batch_idx):
        return train_hybrid_fusion(self, batch)
    
    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch)
        self.log_eval(preds, labels, 'valid')

    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch)

        self.log_eval(preds, labels, 'test')

        return None

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,
                #relative_step=False,
                #scale_parameter=False,
                #warmup_init=False,
            )

        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=self.num_warmup_steps,
                                                    #num_training_steps=self.num_training_steps,
                                                    )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": 'step',
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": 'linear_schedule_with_warmup',
        }
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,
                }

class ViltModel(DefaultModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder, hidden_dropout_prob=self.dropout,
                                                     attention_probs_dropout_prob=self.dropout,
                                                     )

        self.freeze()

        self.classifier = nn.Sequential(nn.Linear(768, 768*2),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        nn.LayerNorm(768*2),
                                        ForwardModuleList(nn.Sequential(nn.Linear(768*2, 768//2),
                                                                        nn.Tanh(),
                                                                        nn.Linear(768//2, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        return self.classifier(self.transformer(*args, **kwargs).last_hidden_state[:, 0])

    def freeze(self):
        freeze_module(self.transformer.embeddings)
        for layer in self.transformer.encoder.layer[:self.frozen]:
            freeze_module(layer)

class CLIPBaseModel(DefaultModel):
    def freeze(self):
        if self.frozen == 'all':
            freeze_module(self.transformer)
        
        elif self.frozen == 'body':
            freeze_module(self.transformer.text_model)
            freeze_module(self.transformer.vision_model)

        else:
            f = float(self.frozen)
            n_text = int(len(self.transformer.text_model.encoder.layers) * f)
            n_vision = int(len(self.transformer.vision_model.encoder.layers) * f)

            freeze_module(self.transformer.text_model.embeddings)
            freeze_module(self.transformer.vision_model.embeddings)

            for layer in self.transformer.text_model.encoder.layers[:n_text]:
                freeze_module(layer)

            for layer in self.transformer.vision_model.encoder.layers[:n_vision]:
                freeze_module(layer)

class CLIPModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,
                                                   )
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        hidden_size = 16
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.LayerNorm(768*4),
                                        nn.Linear(768*4, hidden_size*2),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        nn.LayerNorm(hidden_size*2),
                                        ForwardModuleList(nn.Sequential(nn.Linear(hidden_size*2, hidden_size//2),
                                                                        nn.Tanh(),
                                                                        nn.Linear(hidden_size//2, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds

        dif = torch.abs(img-txt)
        mul = img*txt
        combined = torch.cat([dif, mul, img, txt], dim=-1)

        return self.classifier(combined)

class CLIPv2Model(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 16
        self.projector = nn.Linear(768, proj_size)
        hidden_size = 32
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.LayerNorm(proj_size*4),
                                        nn.Linear(proj_size*4, hidden_size*2),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        nn.LayerNorm(hidden_size*2),
                                        ForwardModuleList(nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                                                        nn.Tanh(),
                                                                        nn.Linear(hidden_size, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        
        img_p = self.projector(img)
        txt_p = self.projector(txt)
        
        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p], dim=-1)

        return self.classifier(combined)

class CLIPCrossAttentionModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 768
        hidden_size = 128

        self.projector = AttentionFeatureMatrixFuser(proj_size, hidden_size, dropout=self.dropout)
        self.classifier = ForwardModuleList(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                          nn.GELU(),
                                                          nn.Linear(hidden_size, i)) for i in self.output_list)

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, \
                   output.text_embeds
        
        attended, *_ = self.projector(img, txt)

        return self.classifier(attended)

class CLIPSimilarityModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 768
        hidden_size = 128

        self.projector = AlignedSimilarityFuser(proj_size, hidden_size)
        self.classifier = ForwardModuleList(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                          nn.GELU(),
                                                          nn.Linear(hidden_size, i)) for i in self.output_list)

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, \
                   output.text_embeds
        
        attended = self.projector(img, txt)

        return self.classifier(attended)

class CLIPSimilarityOnlyModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        self.projector = SimilarityFuser(768)
        self.classifier = ForwardModuleList(nn.Sequential(nn.Linear(3, 3),
                                                          nn.GELU(),
                                                          nn.Linear(3, i)) for i in self.output_list)

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, \
                   output.text_embeds
        
        attended = self.projector(img, txt)

        return self.classifier(attended)

class CLIPTransformerModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = 768
        self.projector = TransformerExtraFuser(input_dim, layers=3, dropout=self.dropout)
        self.classifier = ForwardModuleList([FuseExtraHead(hidden_dim=input_dim, 
                                                        n_outputs=i, 
                                                        n_layers=3, 
                                                        dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.transformer.visual_projection(output.vision_model_output.last_hidden_state), \
                   self.transformer.text_projection(output.text_model_output.last_hidden_state)
        projected_img, projected_txt, projected_diff, projected_mult = self.projector(img, txt, mask=kwargs['attention_mask'])
        token_img, token_txt = img[:, 0], txt[:, 0]

        r_img = token_img+projected_img
        r_txt = token_txt+projected_txt
        r_diff = torch.abs(token_img-token_txt) + projected_diff
        r_mult = token_img*token_txt + projected_mult
        return self.classifier(r_img, r_txt, r_diff, r_mult)

class CLIPWeightedModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        hidden_dim = 128
        self.projector = nn.Sequential(nn.Dropout(self.dropout),
                                       nn.LayerNorm(input_dim),
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.GELU())
        self.classifier = ForwardModuleList([WeightedHead(hidden_dim, 
                                                          hidden_dim, i, 
                                                          self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        
        return self.classifier(img, txt)

class CLIPFuseModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.classifier = ForwardModuleList([FuseHead(hidden_dim=input_dim, 
                                                      n_outputs=i, 
                                                      n_layers=1, 
                                                      dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        return self.classifier(img, txt)

class CLIPReductionModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.classifier = ForwardModuleList([ReductionHead(hidden_dim=input_dim, reduce_dim=32, 
                                                       n_outputs=i, 
                                                       n_layers=1, 
                                                       dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        return self.classifier(img, txt)

class CLIPReductionFuseModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.classifier = ForwardModuleList([ReduceFuseHead(hidden_dim=input_dim, reduce_dim=32, 
                                                            n_outputs=i, 
                                                            n_layers=1, 
                                                            dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        return self.classifier(img, txt)

class CLIPHybridFusionModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.multi_classifier = ForwardModuleList([FuseHead(hidden_dim=input_dim, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])


    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        
        return self.multi_classifier(img, txt), self.vision_classifier(img), self.text_classifier(txt)

class CLIPNormalizeProjectModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        output_dim = 128
        self.projector = NormalizeProject(input_dim, output_dim)
        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=output_dim*4, 
                                                          n_outputs=i, 
                                                          n_layers=3, 
                                                          dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        outs = self.projector(img, txt)
        combined = torch.cat(outs, dim=-1)
        return self.multi_classifier(combined)


class CLIPHybridReductionFusionModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.multi_classifier = ForwardModuleList([ReduceFuseHead(hidden_dim=input_dim, reduce_dim=32,
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])


    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        
        return self.multi_classifier(img, txt), self.vision_classifier(img), self.text_classifier(txt)

class CLIPWeightedFusionModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = self.transformer.config.projection_dim
        self.projector = FFResLayer(input_dim, dropout=self.dropout)
        self.multi_classifier = ForwardModuleList([FuseHead(hidden_dim=input_dim, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=input_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])
        self.attention = ForwardModuleList([FFLayer(input_dim*4, 3, dropout=0) for _ in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.projector(output.image_embeds), \
                   self.projector(output.text_embeds)
        
        attentions = self.attention(torch.cat([img, txt, torch.abs(img-txt), img*txt], -1))

        m_preds = self.multi_classifier(img, txt)
        i_preds = self.vision_classifier(img)
        t_preds = self.text_classifier(txt)

        f_preds = []
        for m, i, t, a in zip(m_preds, i_preds, t_preds, attentions):
            m_a, i_a, t_a = torch.softmax(a, dim=-1).unbind(-1)
            
            f_preds.append(m*m_a.unsqueeze(-1) + i*i_a.unsqueeze(-1) + t*t_a.unsqueeze(-1))


        return f_preds, i_preds, t_preds

class CLIPTransformerModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        self.loss_balancer = None #EMABalancer(len(self.output_list), beta=.1)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = 768
        self.projector = TransformerExtraFuser(input_dim, layers=3, dropout=self.dropout)
        self.classifier = ForwardModuleList([FuseExtraHead(hidden_dim=input_dim, 
                                                        n_outputs=i, 
                                                        n_layers=3, 
                                                        dropout=self.dropout) for i in self.output_list])

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.transformer.visual_projection(output.vision_model_output.last_hidden_state), \
                   self.transformer.text_projection(output.text_model_output.last_hidden_state)
        projected_img, projected_txt, projected_diff, projected_mult = self.projector(img, txt, mask=kwargs['attention_mask'])
        token_img, token_txt = img[:, 0], txt[:, 0]

        r_img = token_img+projected_img
        r_txt = token_txt+projected_txt
        r_diff = torch.abs(token_img-token_txt) + projected_diff
        r_mult = token_img*token_txt + projected_mult
        return self.classifier(r_img, r_txt, r_diff, r_mult)



'''
class ManualModel(DefaultModel):
    def __init__(self,
                 accumulation,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation = accumulation
        self.automatic_optimization = False
        self.loss_balancer = LossBalancer(len(self.output_list), pooling=None)

        self.save_hyperparameters()
    
    def train_dataloader(self):
        if self.semi_supervised:
            self.relabel()

        return self.train_dataloader_obj
    
    def training_step(self, batch, batch_idx):
        encoded_batch, labels=batch
        opts = self.optimizers()
        multitask=len(labels.shape) > 1

        if not multitask:
            labels=labels.unsqueeze(-1)

        x_hat_list=self(**encoded_batch)

        losses=[]
        for x_hat, y, w in zip(x_hat_list,
                               labels.unbind(axis=-1),
                               self.class_weights):
            cw=w.to(self.device).float()
            loss=F.cross_entropy(x_hat, y,
                                   weight = cw,
                                   label_smoothing = .1)
            losses.append(loss)

        losses=self.loss_balancer(torch.stack(losses))
        for i, (o, l) in enumerate(zip(opts, losses.unbind(axis=-1))):
            self.manual_backward(l)
            self.log(f"train_loss_{i}", l)
            o.step()
            o.zero_grad()

        self.log(f"train_loss", torch.nanmean(losses))
        
        return torch.nanmean(losses)
    
    def configure_optimizers(self):
        opts = []
        for _ in range(len(self.output_list)):
            optimizer = Adafactor([
                {"params": self.transformer.parameters(), "lr": self.learning_rate},
                {"params": self.classifier.parameters(
                ), "lr": self.learning_rate*10},
            ],
                lr=self.learning_rate,
                weight_decay=1e-2,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )

            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps,
                                                        )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            }
            opts.append({'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler_config,
                    })
        return opts

class CLIPv3Model(ManualModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 128
        self.projector = nn.Linear(768, proj_size)
        hidden_size = 128
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.LayerNorm(proj_size*4),
                                        nn.Linear(proj_size*4, hidden_size*2),
                                        nn.GELU(),
                                        nn.Dropout(self.dropout),
                                        nn.LayerNorm(hidden_size*2),
                                        ForwardModuleList(nn.Sequential(nn.Linear(hidden_size*2, hidden_size//2),
                                                                        nn.Tanh(),
                                                                        nn.Linear(hidden_size//2, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        
        img_p = self.projector(img)
        txt_p = self.projector(txt)
        
        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p], dim=-1)

        return self.classifier(combined)

    def freeze(self):
        freeze_module(self.transformer.text_model.embeddings)
        freeze_module(self.transformer.vision_model.embeddings)

        for layer in self.transformer.text_model.encoder.layers[:self.frozen]:
            freeze_module(layer)

        for layer in self.transformer.vision_model.encoder.layers[:self.frozen*2]:
            freeze_module(layer)




class CLIPv3Model(DefaultModel):
    def training_step(self, batch, batch_idx):
        encoded_batch, labels=batch

        multitask=len(labels.shape) > 1

        if not multitask:
            labels=labels.unsqueeze(-1)

        img_logits, txt_logits, multi_logits =self(**encoded_batch)

        losses=[]
        for img, txt, multi, y, w in zip(img_logits,
                               txt_logits,
                               multi_logits,
                               labels.unbind(axis=-1),
                               self.class_weights):
            cw=w.to(self.device).float()
            img_loss=F.cross_entropy(img, y,
                                     weight = cw,
                                     label_smoothing = .0)
            txt_loss=F.cross_entropy(txt, y,
                                     weight = cw,
                                     label_smoothing = .0)
            multi_loss=F.cross_entropy(multi, y,
                                     weight = cw,
                                     label_smoothing = .0)

            losses.append(img_loss+txt_loss+multi_loss)

        loss=self.loss_balancer(torch.stack(losses))
        self.log("train_loss", loss)
        return loss

    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,
                                                   )
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 128
        self.projector = nn.Linear(768, proj_size)
        hidden_size = 128
        self.multi_classifier = nn.Sequential(ForwardModuleList(nn.Sequential(nn.Linear(2*sum(self.output_list) + proj_size*4, hidden_size//2),
                                                                              nn.Tanh(),
                                                                              nn.Linear(hidden_size//2, i)) for i in self.output_list))
        self.img_classifier = nn.Sequential(ForwardModuleList(nn.Sequential(nn.Linear(proj_size, hidden_size//2),
                                                                              nn.Tanh(),
                                                                              nn.Linear(hidden_size//2, i)) for i in self.output_list))
        self.txt_classifier = nn.Sequential(ForwardModuleList(nn.Sequential(nn.Linear(proj_size, hidden_size//2),
                                                                              nn.Tanh(),
                                                                              nn.Linear(hidden_size//2, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        
        img_p = self.projector(img)
        txt_p = self.projector(txt)
        
        img_preds = self.img_classifier(img_p)
        txt_preds = self.txt_classifier(txt_p)

        img_preds_cat = torch.cat(img_preds, dim=-1)
        txt_preds_cat = torch.cat(txt_preds, dim=-1)

        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p, img_preds_cat, txt_preds_cat], dim=-1)
        multi_preds = self.multi_classifier(combined)
        
        return img_preds, txt_preds, multi_preds

    def freeze(self):
        freeze_module(self.transformer.text_model.embeddings)
        freeze_module(self.transformer.vision_model.embeddings)

        for layer in self.transformer.text_model.encoder.layers[:self.frozen]:
            freeze_module(layer)

        for layer in self.transformer.vision_model.encoder.layers[:self.frozen*2]:
            freeze_module(layer)

    def configure_optimizers(self):
        optimizer = AdamW([
            {"params": self.transformer.parameters(), "lr": self.learning_rate},
            {"params": self.img_classifier.parameters(), "lr": self.learning_rate*10},
            {"params": self.txt_classifier.parameters(), "lr": self.learning_rate*10},
            {"params": self.multi_classifier.parameters(), "lr": self.learning_rate*10},

        ],
            lr=self.learning_rate,
            weight_decay=1e-2,
            amsgrad=True,
        )

        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=self.num_training_steps,
                                                    )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": 'step',
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": 'linear_schedule_with_warmup',
        }
        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,
                }

    def relabel(self):
        print('Relabelling...')
        old_aug = self.train_dataloader_obj.collate_fn.augment_mode
        self.train_dataloader_obj.collate_fn.augment_mode = 'soft'
        temp_loader = DataLoader(self.original_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 collate_fn=self.train_dataloader_obj.collate_fn,
                                 num_workers=8,
                                 prefetch_factor=4,
        )
        with torch.no_grad():
            new_labels = []
            for encoded_batch, labels in tqdm(temp_loader):
                
                labels = labels.to(self.device)
                encoded_batch = encoded_batch.to(self.device)

                x_hat_list = self(**encoded_batch)[-1]
                preds = []

                for x_hat in x_hat_list:
                    mx, agmx = x_hat.max(-1)
                    high_conf = torch.where(mx > .95, agmx, -100)

                    preds.append(high_conf)

                pseudo_labels = torch.stack(preds, dim=-1)

                new_labels += torch.where(labels == -100,
                                          pseudo_labels,
                                          labels).cpu().numpy().tolist()
        
        self.train_dataloader_obj.collate_fn.augment_mode = old_aug
        self.train_dataloader_obj = DataLoader(AugmentedWrapperDataset(self.original_dataset, new_labels),
                                               batch_size=self.train_dataloader_obj.batch_size,
                                               shuffle=True,
                                               collate_fn=self.train_dataloader_obj.collate_fn,
                                               num_workers = self.train_dataloader_obj.num_workers,
                                               prefetch_factor = self.train_dataloader_obj.prefetch_factor,
                                               pin_memory = self.train_dataloader_obj.pin_memory,
                                    )


class CLIPConvBlock(nn.Module):
    def __init__(self, in_f, out_f, dropout):
        super().__init__()
        self.mdl = nn.Sequential(nn.BatchNorm2d(in_f),
                                    nn.Conv2d(in_f, out_f, kernel_size=3),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.MaxPool2d(2, stride=2),
                                    )
        
    def __call__(self, x):
        return self.mdl(x)


class CLIPConvModel(DefaultModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,
                                                   )
        set_dropout(self.transformer, self.dropout)
        
        self.freeze()
        
        hidden = 256
        self.convnet = nn.Sequential(CLIPConvBlock(1, hidden//4, self.dropout),
                                     CLIPConvBlock(hidden//4, hidden//2, self.dropout),
                                     CLIPConvBlock(hidden//2, hidden, self.dropout),
                                     )
        self.classifier = nn.Sequential(nn.Dropout(self.dropout),
                                        nn.LayerNorm(hidden),
                                        ForwardModuleList(nn.Sequential(nn.Linear(hidden, hidden//2),
                                                                        nn.Tanh(),
                                                                        nn.Linear(hidden//2, i)) for i in self.output_list))

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        bs = img.shape[0]
        x = (img.unsqueeze(2) @ txt.unsqueeze(1)).unsqueeze(1)
        x = self.convnet(x)
        x = x.mean([-1, -2])

        return self.classifier(x)

    def freeze(self):
        freeze_module(self.transformer.text_model.embeddings)
        freeze_module(self.transformer.vision_model.embeddings)

        for layer in self.transformer.text_model.encoder.layers[:self.frozen]:
            freeze_module(layer)

        for layer in self.transformer.vision_model.encoder.layers[:self.frozen*2]:
            freeze_module(layer)


'''
# for openai/clip-vit-large-patch14-336
'''
class FlavaWithCrossModalConvolutionModel(DefaultModel):
    def freeze(self):  # Frozen means the layers that are finetuned
        freeze_module(self.transformer.text_model.embeddings)
        freeze_module(self.transformer.image_model.embeddings)

        for layer in self.transformer.text_model.encoder.layer[-self.frozen:]:
            freeze_module(layer)

        for layer in self.transformer.image_model.encoder.layer[-self.frozen:]:
            freeze_module(layer)

        for layer in self.transformer.multimodal_model.encoder.layer[-self.frozen:]:
            freeze_module(layer)

    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,
                                                     image_config=FlavaImageConfig(hidden_dropout_prob=self.dropout,
                                                                                   attention_probs_dropout_prob=self.dropout,
                                                                                   ),
                                                     text_config=FlavaTextConfig(hidden_dropout_prob=self.dropout,
                                                                                 attention_probs_dropout_prob=self.dropout,
                                                                                 ),
                                                     multimodal_config=FlavaMultimodalConfig(hidden_dropout_prob=self.dropout,
                                                                                             attention_probs_dropout_prob=self.dropout,
                                                                                             ),
                                                     )

        self.freeze()
        self.classifier = nn.Sequential(nn.Linear(768*9, 128),
                                    nn.GELU(),
                                    nn.Dropout(.5),
                                    nn.LayerNorm(128),
                                    ForwardModuleList(nn.Sequential(nn.Linear(128, 32),
                                                                    nn.Tanh(),
                                                                    nn.Linear(32, i)) for i in self.output_list)
                                    )

    def forward(self, attention_mask=None, *args, **kwargs):
        out = self.transformer(*args, **kwargs)
        x = out.image_embeddings[:, 0]
        y = out.text_embeddings[:, 0]
        z = out.multimodal_embeddings[:, 0]
        # x, y, z, torch.abs(x-y), torch.abs(x*y), torch.abs(x-z), torch.abs(x*z), torch.abs(y-z), torch.abs(y*z)
        x = torch.cat((x, y, z,
                       torch.abs(x-y), x*y,
                       torch.abs(x-z), x*z,
                       torch.abs(y-z), y*z,
                       ), dim=-1)
        return self.classifier(x)
'''

'''    def relabel(self):
        print('Relabelling...')
        old_aug = self.train_dataloader_obj.collate_fn.augment_mode
        self.train_dataloader_obj.collate_fn.augment_mode = 'soft'
        temp_loader = DataLoader(self.original_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 collate_fn=self.train_dataloader_obj.collate_fn,
                                 num_workers=8,
                                 prefetch_factor=4,
        )
        
        with torch.no_grad():
            new_labels = []
            for encoded_batch, labels in tqdm(temp_loader):
                
                labels = labels.to(self.device)
                encoded_batch = encoded_batch.to(self.device)

                x_hat_list = self(**encoded_batch)
                preds = []

                for x_hat in x_hat_list:
                    mx, agmx = x_hat.max(-1)
                    high_conf = torch.where(mx > .9, agmx, -100)

                    preds.append(high_conf)

                pseudo_labels = torch.stack(preds, dim=-1)

                new_labels += torch.where(labels == -100,
                                          pseudo_labels,
                                          labels).cpu().numpy().tolist()
        
        self.train_dataloader_obj.collate_fn.augment_mode = old_aug
        self.train_dataloader_obj = DataLoader(AugmentedWrapperDataset(self.original_dataset, new_labels),
                                               batch_size=self.train_dataloader_obj.batch_size,
                                               shuffle=True,
                                               collate_fn=self.train_dataloader_obj.collate_fn,
                                               num_workers = self.train_dataloader_obj.num_workers,
                                               prefetch_factor = self.train_dataloader_obj.prefetch_factor,
                                               pin_memory = self.train_dataloader_obj.pin_memory,
                                    )

    def set_train_dataloader(self, x):
        self.train_dataloader_obj = x
        self.original_dataset = x.dataset

    def train_dataloader(self):
        if self.current_epoch > 1 and self.semi_supervised:
            self.relabel()

        return self.train_dataloader_obj
'''