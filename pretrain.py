from transformers import AutoModel, Adafactor
from transformers import CLIPTextConfig, CLIPVisionConfig
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule_with_warmup
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch.nn.functional as F
import pytorch_lightning as L
import numpy as np

import torch
import einops

from modules import ForwardModuleList, LossBalancer, EMABalancer, AlignedFuser
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import AugmentedWrapperDataset

from torchmetrics import Accuracy, F1Score, AUROC


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
        binary_labels = (torch.max(labels, dim=-1)[1] != 0).float()
        binary_preds = preds[:, 1:].sum(-1) - preds[:, 0]

        for metric, function in self.metrics[key].items():
            function(binary_preds, binary_labels.squeeze())
            self.log(f'{split}_{metric}', function, on_epoch=validating)

    def training_step(self, batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch).squeeze()
        binary_preds = preds[:, 1:].sum(-1) - preds[:, 0]
        binary_labels = (torch.max(labels, dim=-1)[1] != 0).float()
        bin_loss = F.binary_cross_entropy_with_logits(binary_preds,
                                                      binary_labels.squeeze(),
                                                      reduction='none',
                                                      )
        #bcw = torch.tensor([1., .1], device=self.device)
        bcw = torch.tensor([1., .2], device=self.device)
        bin_loss = torch.where(binary_labels==1, bcw[0] * bin_loss, bcw[1] * bin_loss).mean()       
        mul_loss = F.cross_entropy(preds,
                                   torch.softmax(labels, -1),
                                   reduction='mean',
                                   weight=torch.tensor([.2, 1., 1., 1., 1., 1.], device=self.device),
                                   )
        beta = .2
        loss = beta * mul_loss + (1-beta) * bin_loss 

        self.log("train_loss", loss)
        self.log("train_mloss", mul_loss)
        self.log("train_bloss", bin_loss)
        self.log_eval(preds.detach(), labels.detach(), 'train')
        return loss

    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch).squeeze()
        self.log_eval(preds, labels, 'valid')

    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds = self(**encoded_batch).squeeze()

        self.log_eval(preds, labels, 'test')

        return None

    def configure_optimizers(self):
        optimizer = Adafactor([
            {"params": self.parameters(), "lr": self.learning_rate},
            ],
            lr=self.learning_rate,
            weight_decay=1e-2,
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
'''
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