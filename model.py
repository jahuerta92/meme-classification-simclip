from transformers import AutoModel, AdamW
from transformers import get_linear_schedule_with_warmup
from torch import nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch.nn.functional as F
import pytorch_lightning as L

import torch
import numpy as np

from modules import ForwardModuleList

class DefaultModel(L.LightningModule):
    def __init__(self, encoder, class_weights, training_steps, lr=1e-4, frozen=8):
        super().__init__()

        if type(class_weights) is list:
            output_list = [len(cw) for cw in class_weights]
            class_weights_torch = [torch.tensor(cw) for cw in class_weights]
        else:
            output_list = [len(class_weights)]
            class_weights_torch = torch.tensor(class_weights).unsqueeze(0)


        self.transformer = AutoModel.from_pretrained(encoder)
        
        self.lr = lr
        self.num_warmup_steps = 0 #int(0.1*training_steps)
        self.num_training_steps = training_steps
        self.output_list = output_list
        self.class_weights = class_weights_torch
        self.frozen = frozen
        
        self.configure_transformer()

    def freeze(self):
        for param in self.transformer.embeddings.parameters():
            param.requires_grad = False
        
        for layer in self.transformer.encoder.layer[:self.frozen]:
            for param in layer.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        encoded_batch, labels = batch

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(-1)
        
        unstacked_labels = labels.unbind(axis=-1)

        x_hat_list = self(**encoded_batch)

        loss = 0

        l = 1/len(self.output_list)
        for x_hat, y, w in zip (x_hat_list,
                                   unstacked_labels, 
                                   self.class_weights):
            cw = w.to(self.device).float()
            loss += F.cross_entropy(x_hat, y,
                                    weight=cw, 
                                    label_smoothing=.0) * l

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self,  batch, batch_idx):
        encoded_batch, labels = batch

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(-1)
        
        unstacked_labels = labels.unbind(axis=-1)
        
        x_hat_list = self(**encoded_batch)

        loss = 0
        scores = []
        
        l = 1/len(self.output_list)
        for x_hat, y, w in zip (x_hat_list,
                                   unstacked_labels, 
                                   self.class_weights):
            cw = w.to(self.device).float()
            loss += F.cross_entropy(x_hat, y,
                                    weight=cw, 
                                    label_smoothing=.0) * l
            x_p = x_hat.argmax(-1)
            if x_hat.shape[-1] == 2:
                x_hat_alt = x_hat[:, 0]
            else:
                x_hat_alt = x_hat

            score = (f1_score(y.detach().cpu().numpy(),
                             x_p.detach().cpu().numpy(),
                             average='macro',
                             ),
                    accuracy_score(y.detach().cpu().numpy(),
                                   x_p.detach().cpu().numpy(),
                                    ),
                    roc_auc_score(y.detach().cpu().numpy(),
                                  F.softmax(x_hat_alt.detach().cpu().float(), dim=-1).numpy(),
                                  labels=np.arange(x_hat.shape[-1]),
                                  average='macro', multi_class='ovo',
                                  ),
                    )     
            scores.append(score)

        loss /= len(x_hat_list)

        self.log("val_loss", loss)
        for i, s in enumerate(scores):
            f1, acc, auc = s
            self.log(f"valid_{i}_acc", acc)
            self.log(f"valid_{i}_f1", f1)
            self.log(f"valid_{i}_auc", auc)

        if len(scores) > 1:
            f1s, accs, aucs = list(zip(*scores))
            self.log(f"valid_mean_acc", sum(accs)/len(accs))
            self.log(f"valid_mean_f1", sum(f1s)/len(f1s))
            self.log(f"valid_mean_auc", sum(aucs)/len(aucs))
        else:
            f1s, accs, aucs = scores[0]
            self.log(f"valid_mean_acc", accs)
            self.log(f"valid_mean_f1", f1s)
            self.log(f"valid_mean_auc", aucs)

        return loss

    def test_step(self,  batch, batch_idx):
        encoded_batch, labels = batch

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(-1)
        
        unstacked_labels = labels.unbind(axis=-1)
        
        x_hat_list = self(**encoded_batch)

        scores = []
        for x_hat, y in zip (x_hat_list, unstacked_labels):
            x_p = x_hat.argmax(-1)
            if x_hat.shape[-1] == 2:
                x_hat_alt = x_hat[:, 0]
            else:
                x_hat_alt = x_hat

            score = (f1_score(y.detach().cpu().numpy(),
                             x_p.detach().cpu().numpy(),
                             average='macro',
                             ),
                    accuracy_score(y.detach().cpu().numpy(),
                                   x_p.detach().cpu().numpy(),
                                    ),
                    roc_auc_score(y.detach().cpu().numpy(),
                                  F.softmax(x_hat_alt.detach().cpu().float(), dim=-1).numpy(),
                                  labels=np.arange(x_hat.shape[-1]),
                                  average='macro', multi_class='ovo',
                                  ),
                    )     
            scores.append(score)

        for i, s in enumerate(scores):
            f1, acc, auc = s
            self.log(f"test_{i}_acc", acc)
            self.log(f"test_{i}_f1", f1)
            self.log(f"test_{i}_auc", auc)

        if len(scores) > 1:
            f1s, accs, aucs = list(zip(*scores))
            self.log(f"test_mean_acc", sum(accs)/len(accs))
            self.log(f"test_mean_f1", sum(f1s)/len(f1s))
            self.log(f"test_mean_auc", sum(aucs)/len(aucs))
        else:
            f1s, accs, aucs = scores[0]
            self.log(f"test_mean_acc", accs)
            self.log(f"test_mean_f1", f1s)
            self.log(f"test_mean_auc", aucs)

        return None

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.lr,
                          weight_decay=1e-2,
                          correct_bias=True,
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

class ViltModel(DefaultModel):
    def configure_transformer(self):
        self.freeze()
        self.transformer.pooler.dense = nn.Sequential(nn.Linear(768, 768),
                                                      nn.GELU(),
                                                      nn.Dropout(.5),
                                                      )
        self.transformer.pooler.activation = ForwardModuleList(nn.Sequential(nn.LayerNorm(768),
                                                                             nn.Linear(768, 386),
                                                                             nn.Tanh(),
                                                                             nn.LayerNorm(386),
                                                                             nn.Linear(386, i)) for i in self.output_list)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs).pooler_output

class Layoutlmv3Model(DefaultModel):
    def configure_transformer(self):
        self.freeze()
        self.pooling_module = nn.Sequential(nn.Linear(1024, 1024),
                                            nn.LayerNorm(1024), 
                                            nn.GELU(),
                                            nn.Dropout(.5),
                                            ForwardModuleList(nn.Sequential(nn.Linear(1024, 512),
                                                                             nn.LayerNorm(512),
                                                                             nn.Tanh(),
                                                                             nn.Linear(512, i)) for i in self.output_list),
                                            )

    def forward(self, *args, **kwargs):
        x = self.transformer(*args, **kwargs).last_hidden_state[:, 0]
        return self.pooling_module(x)
