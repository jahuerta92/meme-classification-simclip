import pytorch_lightning as L
import torch.nn.functional as F
import torch
from torchmetrics import MetricCollection
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

from torch.optim import AdamW
from transformers import AutoModel

from torch import nn

from modules import BatchnNormProject, FFHead, FFLayer, FFResLayer, ForwardModuleList, NormalizeProject

def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False

class DefaultModel(L.LightningModule):
    def __init__(self, 
                 class_weights, 
                 training_steps,
                 dropout=.1,
                 lr=1e-4,
                 warmup_p=.1,
                 frozen='base',
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.frozen = frozen
        self.dropout = dropout
        self.learning_rate = lr
        self.num_training_steps = training_steps
        self.num_warmup_steps = int(warmup_p*training_steps)
        self.eps = 1e-8
        self.loss_functions = [nn.CrossEntropyLoss(weight=torch.tensor(cw).float()) for cw in class_weights]
        self.output_list = [len(cw) for cw in class_weights]
        #self.name
        self.define_metrics()

    def define_metrics(self):
        self.metrics = {'train':[], 
                        'valid':[], 
                        'test':[]}

        for i, n in enumerate(self.output_list):
            metrics = MetricCollection([MulticlassAccuracy(n), 
                                         MulticlassAUROC(n), 
                                         MulticlassF1Score(n)])
            for k in self.metrics.keys():
                self.metrics[k].append(metrics.clone(prefix=f'{k}/target_{i}_'))

    def update_metrics(self, preds, labels, split='valid'):
        if len(labels.shape) <= 1:
            labels=labels.unsqueeze(-1)
        
        for m, l, p in zip(self.metrics[split], labels.unbind(-1), preds):
            m.update(p.cpu(), l.cpu())
            
    def compute_metrics(self, split='valid'):
        for m in self.metrics[split]:
            output = m.compute()
            self.log_dict(output)
            m.reset()

    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds, *_ = self(*encoded_batch)
        self.update_metrics(preds, labels, 'valid')
        return None

    def on_validation_epoch_end(self):
        self.compute_metrics('valid')
        
    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds, *_ = self(*encoded_batch)
        self.update_metrics(preds, labels, 'test')
        return None

    def on_test_epoch_end(self):
        self.compute_metrics('test')
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-4,

            )

        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=self.num_warmup_steps,
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

class BaseClipModel(DefaultModel):
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal = AutoModel.from_pretrained(models['multi'])
        if self.frozen == 'all':
            freeze_module(self.multimodal)
        
        elif self.frozen == 'body':
            freeze_module(self.multimodal.text_model)
            freeze_module(self.multimodal.vision_model)

        else:
            f = float(self.frozen)
            n_text = int(len(self.multimodal.text_model.encoder.layers) * f)
            n_vision = int(len(self.multimodal.vision_model.encoder.layers) * f)

            freeze_module(self.multimodal.text_model.embeddings)
            freeze_module(self.multimodal.vision_model.embeddings)

            for layer in self.multimodal.text_model.encoder.layers[:n_text]:
                freeze_module(layer)

            for layer in self.multimodal.vision_model.encoder.layers[:n_vision]:
                freeze_module(layer)

    def training_step(self,  batch, batch_idx):
        (multimodal, ), labels = batch
        preds, *_ = self(multimodal)
        if len(labels.shape) <= 1:
            labels=labels.unsqueeze(-1)

        losses = []
        for m_p, y, l in zip(preds,
                             labels.unbind(axis=-1),
                             self.loss_functions):
            fnc = l.to(self.device)

            m_loss = fnc(m_p, y)
            losses.append(m_loss)

        loss=torch.nanmean(torch.stack(losses))
        with torch.no_grad():
            self.log('train/loss', loss)
            self.update_metrics(preds, labels, 'train')
            if batch_idx%8 == 0:
                self.compute_metrics('train')
        
        return loss
    
class ClipNormalizeProjectModel(BaseClipModel):
    NAME = 'ClipNormalizeProjectModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.multimodal.config.projection_dim
        output_dim = 128
        self.projector = NormalizeProject(input_dim, output_dim)
        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=output_dim*4, 
                                                          n_outputs=i, 
                                                          n_layers=3, 
                                                          dropout=self.dropout) for i in self.output_list])

    def forward(self, x):
        output = self.multimodal(**x)
        img, txt = output.image_embeds, output.text_embeds
        outs = self.projector(img, txt)
        combined = torch.cat(outs, dim=-1)
        return self.multi_classifier(combined), None
    
class ClipNormModel(BaseClipModel):
    NAME = 'ClipNormModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.multimodal.config.projection_dim
        output_dim = 1024
        self.projector_img = nn.Linear(input_dim, output_dim)
        self.projector_text = nn.Linear(input_dim, output_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=output_dim*4, 
                                                          n_outputs=i, 
                                                          n_layers=3, 
                                                          dropout=self.dropout) for i in self.output_list])

    def forward(self, x):
        output = self.multimodal(**x)
        img, txt = self.projector_img(F.normalize(output.image_embeds,-1)), self.projector_text(F.normalize(output.text_embeds,-1))
        combined = torch.cat([img, txt, torch.abs(img-txt), img*txt], dim=-1)
        return self.multi_classifier(combined), None
  
class HatecliperModel(BaseClipModel):
    NAME = 'HatecliperModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.multimodal.config.projection_dim
        output_dim = 1024
        self.projector_img = nn.Linear(input_dim, output_dim)
        self.projector_text = nn.Linear(input_dim, output_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=output_dim, 
                                                          n_outputs=i, 
                                                          n_layers=3, 
                                                          dropout=self.dropout,
                                                          residual=False,
                                                          norm=False) for i in self.output_list])

    def forward(self, x):
        output = self.multimodal(**x)
        img, txt = self.projector_img(F.gelu(output.image_embeds)), self.projector_text(F.gelu(output.text_embeds))
        img, txt = F.normalize(img,-1), F.normalize(txt,-1)

        return self.multi_classifier(img*txt), None
    
class ClipBatchNormModel(BaseClipModel):
    NAME = 'ClipBatchNormModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = self.multimodal.config.projection_dim
        output_dim = 1024
        self.projector_img = BatchnNormProject(input_dim, output_dim)
        self.projector_text = BatchnNormProject(input_dim, output_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=output_dim*4, 
                                                          n_outputs=i, 
                                                          n_layers=3, 
                                                          dropout=self.dropout) for i in self.output_list])

    def forward(self, x):
        output = self.multimodal(**x)
        img, txt = self.projector_img(output.image_embeds), self.projector_text(output.text_embeds)
        combined = torch.cat([img, txt, torch.abs(img-txt), img*txt], dim=-1)
        return self.multi_classifier(combined), None

class ClipBaselineModel(BaseClipModel):
    NAME = 'ClipBaselineModel'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proj_size = 16
        input_dim = self.multimodal.config.projection_dim
        self.projector = nn.Linear(input_dim, proj_size)
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

    def forward(self, x):
        output = self.multimodal(**x)
        img, txt = output.image_embeds, output.text_embeds
        
        img_p = self.projector(img)
        txt_p = self.projector(txt)
        
        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p], dim=-1)

        return self.classifier(combined), None

class BaseViltXlmtVitModel(DefaultModel):
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal = AutoModel.from_pretrained(models['multi'], hidden_dropout_prob=self.dropout, attention_probs_dropout_prob=self.dropout) # dandelin/vilt-b32-mlm
        self.vision = AutoModel.from_pretrained(models['vision'], hidden_dropout_prob=self.dropout, attention_probs_dropout_prob=self.dropout) # jayanta/vit-base-patch16-224-FV-20epochs-finetuned-memes
        self.text = AutoModel.from_pretrained(models['text'], hidden_dropout_prob=self.dropout, attention_probs_dropout_prob=self.dropout) # cardiffnlp/twitter-xlm-roberta-base

        if self.frozen == 'all':
            freeze_module(self.multimodal)
            freeze_module(self.vision)
            freeze_module(self.text)

        elif self.frozen == 'body':
            freeze_module(self.multimodal)
            freeze_module(self.vision)
            freeze_module(self.text)

        else:
            f = float(self.frozen)
            n_text = int(len(self.text.encoder.layer) * f)
            n_vision = int(len(self.vision.encoder.layer) * f)
            n_multi = int(len(self.multimodal.encoder.layer) * f)

            freeze_module(self.multimodal.embeddings)
            freeze_module(self.text.embeddings)
            freeze_module(self.vision.embeddings)

            for layer in self.text.encoder.layer[:n_text]:
                freeze_module(layer)

            for layer in self.vision.encoder.layer[:n_vision]:
                freeze_module(layer)

            for layer in self.multimodal.encoder.layer[:n_multi]:
                freeze_module(layer)

    def training_step(self,  batch, batch_idx):
        (multimodal, vision, language), labels = batch
        preds, multi, image, text = self(multimodal, vision, language)
        if len(labels.shape) <= 1:
            labels=labels.unsqueeze(-1)

        losses = []
        for a_p, m_p, i_p, t_p, y, l in zip(preds, multi, image, text,
                                            labels.unbind(axis=-1),
                                            self.loss_functions):
            fnc = l.to(self.device)

            m_loss = fnc(m_p, y)
            i_loss = fnc(i_p, y)
            t_loss = fnc(t_p, y)
            a_loss = fnc(a_p, y)
            loss = a_loss + m_loss + i_loss + t_loss
            losses.append(loss)

        loss=torch.nanmean(torch.stack(losses))
        with torch.no_grad():
            self.log('train/loss',loss)
            self.update_metrics(preds, labels, 'train')
            if batch_idx%8 == 0:
                self.compute_metrics('train')
        
        return loss

class ViltXlmtVitModel(BaseViltXlmtVitModel):
    NAME = 'ViltXlmtVitModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        m_dim = self.multimodal.config.hidden_size
        i_dim = self.vision.config.hidden_size
        t_dim = self.text.config.hidden_size
        f_dim = m_dim + i_dim + t_dim
        self.projection = FFResLayer(in_dim=m_dim, dropout=self.dropout)
        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=m_dim, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=i_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=t_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])

        self.full_classifier = ForwardModuleList([FFHead(hidden_dim=f_dim, 
                                                         n_outputs=i, 
                                                         n_layers=3, 
                                                         dropout=self.dropout) for i in self.output_list])

    def forward(self, m, i, t):
        multi_output = self.multimodal(**m)
        vision_output = self.vision(**i)
        text_output = self.text(**t)

        multi_cls = self.projection(multi_output.last_hidden_state[:, 0])
        vision_cls = self.projection(vision_output.last_hidden_state[:, 0])
        text_cls = self.projection(text_output.last_hidden_state[:, 0])

        multi_pred = self.multi_classifier(multi_cls)
        vision_pred = self.vision_classifier(vision_cls)
        text_pred = self.text_classifier(text_cls)

        all_cls = torch.cat([multi_cls, vision_cls, text_cls], axis=-1)
        all_pred = self.full_classifier(all_cls)

        return all_pred, multi_pred, vision_pred, text_pred

class BaseClipXlmtVitModel(DefaultModel):
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal = AutoModel.from_pretrained(models['multi']) # openai/clip-vit-large-patch14-336
        self.vision = AutoModel.from_pretrained(models['vision'], hidden_dropout_prob=self.dropout, attention_probs_dropout_prob=self.dropout) # jayanta/vit-base-patch16-224-FV-20epochs-finetuned-memes
        self.text = AutoModel.from_pretrained(models['text'], hidden_dropout_prob=self.dropout, attention_probs_dropout_prob=self.dropout) # cardiffnlp/twitter-xlm-roberta-base

        if self.frozen == 'all':
            freeze_module(self.multimodal)
            freeze_module(self.vision)
            freeze_module(self.text)

        elif self.frozen == 'body':
            freeze_module(self.multimodal.text_model)
            freeze_module(self.multimodal.vision_model)
            freeze_module(self.vision)
            freeze_module(self.text)

        else:
            f = float(self.frozen)
            n_text = int(len(self.text.encoder.layer) * f)
            n_vision = int(len(self.vision.encoder.layer) * f)
            n_mt = int(len(self.multimodal.text_model.encoder.layers) * f)
            n_mv = int(len(self.multimodal.vision_model.encoder.layers) * f)

            freeze_module(self.multimodal.text_model.embeddings)
            freeze_module(self.multimodal.vision_model.embeddings)
            freeze_module(self.text.embeddings)
            freeze_module(self.vision.embeddings)

            for layer in self.text.encoder.layer[:n_text]:
                freeze_module(layer)

            for layer in self.vision.encoder.layer[:n_vision]:
                freeze_module(layer)

            for layer in self.multimodal.text_model.encoder.layers[:n_mt]:
                freeze_module(layer)

            for layer in self.multimodal.vision_model.encoder.layers[:n_mv]:
                freeze_module(layer)

    def training_step(self,  batch, batch_idx):
        (multimodal, vision, language), labels = batch
        preds, multi, image, text = self(multimodal, vision, language)
        if len(labels.shape) <= 1:
            labels=labels.unsqueeze(-1)

        losses = []
        for a_p, m_p, i_p, t_p, y, l in zip(preds, multi, image, text,
                                            labels.unbind(axis=-1),
                                            self.loss_functions):
            fnc = l.to(self.device)

            m_loss = fnc(m_p, y)
            i_loss = fnc(i_p, y)
            t_loss = fnc(t_p, y)
            a_loss = fnc(a_p, y)
            loss = a_loss + m_loss + i_loss + t_loss
            losses.append(loss)

        loss=torch.nanmean(torch.stack(losses))
        with torch.no_grad():
            self.log('train/loss',loss)
            self.update_metrics(preds, labels, 'train')
            if batch_idx%8 == 0:
                self.compute_metrics('train')
        
        return loss

class NormClipXlmtVitModel(BaseClipXlmtVitModel):
    NAME = 'NormClipXlmtVitModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        m_dim = self.multimodal.config.projection_dim
        i_dim = self.vision.config.hidden_size
        t_dim = self.text.config.hidden_size
        common_dim = 1024
        f_dim = common_dim*4

        self.projection_multi_vision = nn.Linear(m_dim, common_dim)
        self.projection_multi_text = nn.Linear(m_dim, common_dim)
        self.projection_vision = nn.Linear(i_dim, common_dim)
        self.projection_text = nn.Linear(t_dim, common_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim*2, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])
        self.full_classifier = ForwardModuleList([FFHead(hidden_dim=f_dim, 
                                                         n_outputs=i, 
                                                         n_layers=3, 
                                                         dropout=self.dropout) for i in self.output_list])

    def forward(self, m, i, t):
        multi_output = self.multimodal(output_hidden_states=True, **m)
        vision_output = self.vision(output_hidden_states=True, **i)
        text_output = self.text(output_hidden_states=True, **t)

        multi_v_cls = F.normalize(self.projection_multi_vision(F.gelu(multi_output.image_embeds),), dim=-1)
        multi_t_cls = F.normalize(self.projection_multi_text(F.gelu(multi_output.text_embeds)), dim=-1)
        vision_cls = F.normalize(self.projection_vision(F.gelu(vision_output.last_hidden_state[:, 0])), dim=-1)
        text_cls = F.normalize(self.projection_text(F.gelu(text_output.last_hidden_state[:, 0])), dim=-1)

        multi = torch.cat([multi_v_cls * multi_t_cls, torch.abs(multi_t_cls - multi_v_cls)], -1)

        multi_pred = self.multi_classifier(multi)
        vision_pred = self.vision_classifier(vision_cls)
        text_pred = self.text_classifier(text_cls)

        all_cls = torch.cat([vision_cls, text_cls, multi], axis=-1)
        all_pred = self.full_classifier(all_cls)

        return all_pred, multi_pred, vision_pred, text_pred

class BatchNormClipXlmtVitModel(BaseClipXlmtVitModel):
    NAME = 'BatchNormClipXlmtVitModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        m_dim = self.multimodal.config.projection_dim
        i_dim = self.vision.config.hidden_size
        t_dim = self.text.config.hidden_size
        common_dim = 128
        f_dim = common_dim*4

        self.projection_multi_vision = BatchNormClipXlmtVitModel(m_dim, common_dim)
        self.projection_multi_text = BatchNormClipXlmtVitModel(m_dim, common_dim)
        self.projection_vision = BatchNormClipXlmtVitModel(i_dim, common_dim)
        self.projection_text = BatchNormClipXlmtVitModel(t_dim, common_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim*2, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])
        self.full_classifier = ForwardModuleList([FFHead(hidden_dim=f_dim, 
                                                         n_outputs=i, 
                                                         n_layers=3, 
                                                         dropout=self.dropout) for i in self.output_list])

    def forward(self, m, i, t):
        multi_output = self.multimodal(output_hidden_states=True, **m)
        vision_output = self.vision(output_hidden_states=True, **i)
        text_output = self.text(output_hidden_states=True, **t)

        multi_v_cls = self.projection_multi_vision(multi_output.image_embeds)
        multi_t_cls = self.projection_multi_text(multi_output.text_embeds)
        vision_cls = self.projection_vision(vision_output.last_hidden_state[:, 0])
        text_cls = self.projection_text(text_output.last_hidden_state[:, 0])

        multi = torch.cat([multi_v_cls * multi_t_cls, torch.abs(multi_t_cls - multi_v_cls)], -1)

        multi_pred = self.multi_classifier(multi)
        vision_pred = self.vision_classifier(vision_cls)
        text_pred = self.text_classifier(text_cls)

        all_cls = torch.cat([vision_cls, text_cls, multi], axis=-1)
        all_pred = self.full_classifier(all_cls)

        return all_pred, multi_pred, vision_pred, text_pred



class PostNormClipXlmtVitModel(BaseClipXlmtVitModel):
    NAME = 'PostNormClipXlmtVitModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        m_dim = self.multimodal.config.projection_dim
        common_dim = 128
        f_dim = common_dim*4
        
        self.projector = NormalizeProject(m_dim, common_dim)

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim*2, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=common_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])

        self.full_classifier = ForwardModuleList([FFHead(hidden_dim=f_dim, 
                                                         n_outputs=i, 
                                                         n_layers=3, 
                                                         dropout=self.dropout) for i in self.output_list])

    def forward(self, m, i, t):
        multi_output = self.multimodal(**m)
        vision_output = self.vision(**i)
        text_output = self.text(**t)

        mvcls = multi_output.image_embeds
        mtcls = multi_output.text_embeds
        vcls = vision_output.last_hidden_state[:, 0]
        tcls = text_output.last_hidden_state[:, 0]        
        
        vis, txt, prod, diff = self.projector(vcls, tcls, mvcls * mtcls, torch.abs(mvcls - mtcls))
        
        multi = torch.cat([prod, diff], -1)

        multi_pred = self.multi_classifier(multi)
        vision_pred = self.vision_classifier(vis)
        text_pred = self.text_classifier(txt)

        all_cls = torch.cat([vis, txt, multi], axis=-1)
        all_pred = self.full_classifier(all_cls)

        return all_pred, multi_pred, vision_pred, text_pred

class ClipXlmtVitModel(BaseClipXlmtVitModel):
    NAME = 'ClipXlmtVitModel'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        m_dim = self.multimodal.config.projection_dim
        i_dim = self.vision.config.hidden_size
        t_dim = self.text.config.hidden_size
        f_dim = m_dim*2 + i_dim + t_dim

        self.multi_classifier = ForwardModuleList([FFHead(hidden_dim=m_dim*2, 
                                                   n_outputs=i, 
                                                   n_layers=1, 
                                                   dropout=self.dropout) for i in self.output_list])
        self.text_classifier = ForwardModuleList([FFHead(hidden_dim=i_dim, 
                                                         n_outputs=i, 
                                                         n_layers=1, 
                                                         dropout=self.dropout) for i in self.output_list])
        self.vision_classifier = ForwardModuleList([FFHead(hidden_dim=t_dim, 
                                                           n_outputs=i, 
                                                           n_layers=1, 
                                                           dropout=self.dropout) for i in self.output_list])

        self.full_classifier = ForwardModuleList([FFHead(hidden_dim=f_dim, 
                                                         n_outputs=i, 
                                                         n_layers=3, 
                                                         dropout=self.dropout) for i in self.output_list])

    def forward(self, m, i, t):
        multi_output = self.multimodal(**m)
        vision_output = self.vision(**i)
        text_output = self.text(**t)

        multi_v_cls = multi_output.image_embeds
        multi_t_cls = multi_output.text_embeds
        vision_cls = vision_output.last_hidden_state[:, 0]
        text_cls = text_output.last_hidden_state[:, 0]

        multi = torch.cat([multi_v_cls * multi_t_cls, torch.abs(multi_t_cls - multi_v_cls)], -1)

        multi_pred = self.multi_classifier(multi)
        vision_pred = self.vision_classifier(vision_cls)
        text_pred = self.text_classifier(text_cls)

        all_cls = torch.cat([vision_cls, text_cls, multi], axis=-1)
        all_pred = self.full_classifier(all_cls)

        return all_pred, multi_pred, vision_pred, text_pred

