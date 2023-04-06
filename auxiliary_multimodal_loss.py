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

from modules import ForwardModuleList, LossBalancer, EMABalancer, AlignedFuser
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import AugmentedWrapperDataset

from torchmetrics import Accuracy, F1Score


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


class AuxiliaryModel(L.LightningModule):
    def __init__(self, encoder, 
                 n_outputs,
                 training_steps,
                 dropout=.5,
                 lr=1e-4,
                 tie_layers=[2, 4, 6, 8, 10],
                 dampen = [0.1, 0.2, 0.4, 0.6, 0.8],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.learning_rate = lr
        self.num_warmup_steps = int(0.1*training_steps)
        self.num_training_steps = training_steps
        self.dropout = dropout
        self.encoder = encoder
        self.n_outputs = n_outputs
        self.tie_layers = tie_layers
        self.dampen = dampen
        self.eps = 1e-8

        self.configure_transformer()
        self.configure_metrics()

    def configure_metrics(self):
        metric_dict = {}
        for split in ['_train', '_valid', '_test']:
            metric_dict[split] = nn.ModuleDict({
                'binary_f1': F1Score('binary'),
                'binary_acc': Accuracy('binary')
            })
        self.metrics = nn.ModuleDict(metric_dict)

    def log_eval(self, preds, labels, split='valid'):
        key = '_' + split

        binary_labels = torch.where(labels[:, 0] > 1, 0., 1.).float().squeeze()

        self.metrics[key]['binary_acc'](preds, binary_labels)
        self.metrics[key]['binary_f1'](preds, binary_labels)
        validating = split in {'valid', 'test'}

        self.log(f"{split}_binary_acc", self.metrics[key]['binary_acc'], on_epoch=validating)
        self.log(f"{split}_binary_f1", self.metrics[key]['binary_f1'], on_epoch=validating)

    def training_step(self, batch, batch_idx):
        encoded_batch, labels = batch
        last_preds, *other_preds = self(**encoded_batch)
        binary_labels = 1. - (labels[:, 0]/3.)
        loss = F.binary_cross_entropy_with_logits(last_preds,
                                                  binary_labels,
                                                  reduction='mean',
                                                  pos_weight=torch.tensor([3.], device=self.device))
        for d, pred in zip(self.dampen, other_preds):
            loss += F.binary_cross_entropy_with_logits(pred,
                                                       binary_labels,
                                                       reduction='mean',
                                                       pos_weight=torch.tensor([3.], device=self.device)) * d        
        self.log("train_loss", loss)

        self.log_eval(last_preds.detach(), labels.detach(), 'train')
        return loss

    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch

        binary_preds, *_ = self(**encoded_batch)
        binary_labels = 1-(labels[:, 0]/3)
        loss = F.binary_cross_entropy_with_logits(binary_preds,
                                                  binary_labels,
                                                  reduction='mean')        
        
        self.log("val_loss", loss)
        self.log_eval(binary_preds, labels, 'valid')
        return loss

    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch
        preds, *_ = self(**encoded_batch)
        self.log_eval(preds, labels, 'test')

        return None

    def configure_optimizers(self):
        optimizer = Adafactor([{"params": self.parameters(), "lr": self.learning_rate},
            #{"params": self.transformer.parameters(), "lr": self.learning_rate},
            {"params": self.fusers.parameters(), "lr": self.learning_rate*10},
            {"params": self.classifiers.parameters(), "lr": self.learning_rate*10},
            ],
            lr=self.learning_rate,
            weight_decay=1e-2,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                      num_warmup_steps=self.num_warmup_steps,
                                                      # num_training_steps=self.num_training_steps,
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


class CLIPAuxiliaryModel(AuxiliaryModel):
    def configure_transformer(self):
        self.transformer = AutoModel.from_pretrained(self.encoder,)
        
        set_dropout(self.transformer, self.dropout)

        proj_size = 128
        hidden_size = 128
        self.fusers = nn.ModuleList(AlignedFuser(1024, 768, proj_size, hidden_size) for _ in range(len(self.tie_layers)+1))
        self.classifiers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(len(self.tie_layers)+1)])
        

    def forward(self, *args, **kwargs):
        output = self.transformer(output_hidden_states=True, *args, **kwargs)
        img_states = output.vision_model_output.hidden_states
        txt_states = output.text_model_output.hidden_states
        inputs = [(img_states[-1][:, 0], txt_states[-1][:, 0])] + [(img_states[layer*2][:, 0], txt_states[layer][:, 0]) for layer in self.tie_layers]
        return [clas(fuse(*x)).squeeze() for x, fuse, clas in zip(inputs, self.fusers, self.classifiers)]


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