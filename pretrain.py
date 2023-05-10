from transformers import AutoModel, Adafactor
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
from torch import nn

import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np

import torch
import einops

from modules import FuseExtraHead, TransformerExtraFuser

from torchmetrics import Accuracy, F1Score, AUROC
from losses import NCEandRCE as LOSS

def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


class PretrainModel(L.LightningModule):
    def __init__(self, encoder, 
                 n_outputs,
                 training_steps,
                 dropout=.5,
                 lr=1e-4,
                 frozen=8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.learning_rate = lr
        self.num_warmup_steps = int(0.1*training_steps)
        self.num_training_steps = training_steps
        self.frozen = frozen
        self.dropout = dropout
        self.encoder = encoder
        self.n_outputs = n_outputs
        self.eps = 1e-8
        
        self.loss = LOSS(num_classes=self.n_outputs, alpha=5., beta=.1, class_weights=torch.tensor([0.23458736,  
                                                                                                     1.22896955,  
                                                                                                     3.48044125,  
                                                                                                     4.78477299, 
                                                                                                     30.48967285,
                                                                                                     2.53554996]))
        '''self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.23458736,  
                                                             1.22896955,  
                                                             3.48044125,  
                                                             4.78477299, 
                                                             30.48967285,
                                                             2.53554996]))'''
        
        self.configure_transformer()
        self.configure_metrics()

    def configure_metrics(self):
        metric_dict = {}
        for split in ['_train', '_valid', '_test']:
            metric_dict[split] = nn.ModuleDict({
                'binary_f1': F1Score('binary', average='macro'),
                'binary_acc': Accuracy('binary'),
                'binary_auroc': AUROC("binary")
            })
        self.metrics = nn.ModuleDict(metric_dict)

    def log_eval(self, preds, labels, split='valid'):
        key = '_' + split

        validating = split in {'valid', 'test'}
        binary_labels = (torch.max(labels, dim=-1)[1] != 0).long()
        hate_ = torch.max(preds[:,1:], dim=-1)[0]
        binary_preds = torch.softmax(torch.stack([preds[:, 0], hate_], dim=-1), -1)[:, 0]

        for metric, function in self.metrics[key].items():
            function.update(binary_preds, binary_labels.squeeze())
            self.log(f'{split}/{metric}', function, on_epoch=validating)

    def training_step(self, batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch)
        _labels = torch.softmax(labels, -1)
        loss = self.loss.to(self.device)(preds, _labels)
        self.log('train/loss', loss)
        self.log_eval(preds, labels, 'train')
        return loss

    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch)
        self.log_eval(preds, labels, 'valid')

    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch)
        self.log_eval(preds, labels, 'test')

    def configure_optimizers(self):
        optimizer = Adafactor([
            {"params": self.parameters(), "lr": self.learning_rate},
            ],
            lr=self.learning_rate,
            weight_decay=1e-4,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=self.num_warmup_steps,
                                                      #                 num_training_steps=self.num_training_steps,
                                                      #                 num_cycles=4
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

class CLIPBaseModel(PretrainModel):
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

class DualEncoderBaseModel(PretrainModel):
    def freeze(self):
        if self.frozen == 'all':
            freeze_module(self.text_model)
            freeze_module(self.vision_model)
        
        elif self.frozen == 'body':
            freeze_module(self.text_model)
            freeze_module(self.vision_model)

        else:
            f = float(self.frozen)
            n_text = int(len(self.text_model.encoder.layers) * f)
            n_vision = int(len(self.vision_model.encoder.layers) * f)

            freeze_module(self.text_model.embeddings)
            freeze_module(self.vision_model.embeddings)

            for layer in self.text_model.encoder.layers[:n_text]:
                freeze_module(layer)

            for layer in self.vision_model.encoder.layers[:n_vision]:
                freeze_module(layer)

class DualEncoderRobustModel(DualEncoderBaseModel):
    def configure_transformer(self):
        #jayanta/vit-base-patch16-224-FV2-finetuned-memes /// 768
        self.vision_model = AutoModel.from_pretrained(self.encoder['vision'],)
        #cardiffnlp/twitter-xlm-roberta-base /// 768
        self.text_model = AutoModel.from_pretrained(self.encoder['text'],)
                 
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = 768
        self.text_projection = nn.Linear(input_dim, input_dim)
        self.visual_projection = nn.Linear(input_dim, input_dim)

        self.projector = TransformerExtraFuser(input_dim, layers=6, dropout=self.dropout)
        self.classifier = FuseExtraHead(hidden_dim=input_dim, 
                                        n_outputs=self.n_outputs, 
                                        n_layers=3, 
                                        dropout=self.dropout)
    
    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.transformer.visual_projection(output.vision_model_output.last_hidden_state), \
                   self.transformer.text_projection(output.text_model_output.last_hidden_state)
        
        projected_img, projected_txt, projected_diff, projected_mult = self.projector(img, 
                                                                                      txt, 
                                                                                      mask=kwargs['attention_mask'])
        
        token_img, token_txt = img[:, 0], txt[:, 0]

        r_img = token_img+projected_img
        r_txt = token_txt+projected_txt
        r_diff = torch.abs(token_img-token_txt) + projected_diff
        r_mult = token_img*token_txt + projected_mult

        return self.classifier(r_img, r_txt, r_diff, r_mult)

class CLIPRobustModel(CLIPBaseModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
                                  
        set_dropout(self.transformer, self.dropout)

        self.freeze()

        input_dim = 768
        self.projector = TransformerExtraFuser(input_dim, layers=6, dropout=self.dropout)
        self.classifier = FuseExtraHead(hidden_dim=input_dim, 
                                        n_outputs=self.n_outputs, 
                                        n_layers=3, 
                                        dropout=self.dropout)
    
    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = self.transformer.visual_projection(output.vision_model_output.last_hidden_state), \
                   self.transformer.text_projection(output.text_model_output.last_hidden_state)
        
        projected_img, projected_txt, projected_diff, projected_mult = self.projector(img, 
                                                                                      txt, 
                                                                                      mask=kwargs['attention_mask'])
        
        token_img, token_txt = img[:, 0], txt[:, 0]

        r_img = token_img+projected_img
        r_txt = token_txt+projected_txt
        r_diff = torch.abs(token_img-token_txt) + projected_diff
        r_mult = token_img*token_txt + projected_mult

        return self.classifier(r_img, r_txt, r_diff, r_mult)
    
'''
class CLIPPretrainModel(PretrainModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder)

        set_dropout(self.transformer, self.dropout)

        self.freeze()

        proj_size = 1024 
        hidden_size = 1024
        self.projector = AlignedFuser(1280, proj_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds
        out, _, _ = self.projector(img, txt)
        return self.classifier(out)

    def freeze(self):
        freeze_module(self.transformer.text_model.embeddings)
        freeze_module(self.transformer.vision_model.embeddings)

        for layer in self.transformer.text_model.encoder.layers[:self.frozen]:
            freeze_module(layer)

        for layer in self.transformer.vision_model.encoder.layers[:self.frozen*2]:
            freeze_module(layer)

class ALIGNPretrainModel(PretrainModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)

        set_dropout(self.transformer, self.dropout)

        proj_size = 128
        self.projector = nn.Linear(640, proj_size)
        hidden_size = 128
        self.classifier = nn.Sequential(nn.Linear(proj_size*4, hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(hidden_size, 1),
        )

    def forward(self, *args, **kwargs):
        output = self.transformer(*args, **kwargs)
        img, txt = output.image_embeds, output.text_embeds

        img_p = self.projector(img)
        txt_p = self.projector(txt)

        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p], dim=-1)

        return self.classifier(combined)'''