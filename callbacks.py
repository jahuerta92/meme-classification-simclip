
from pytorch_lightning.callbacks import Callback
from dataset import AugmentedWrapperDataset
from copy import deepcopy
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch
from tqdm.auto import tqdm
import numpy as np

def unfreeze(layer):
    for param in layer.parameters():
        param.requires_grad = True


class UnfreezeCallback(Callback):
    def __init__(self, unfreeze_at_epoch=5):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.epoch_counter = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # When `current_epoch` is 10, feature_extractor will start training.
        if self.epoch_counter == (self._unfreeze_at_epoch - 1):
            print('Enable all layers')
            for param in pl_module.parameters():
                param.requires_grad = True
        self.epoch_counter+=1
         
class UnfreezeCLIPCallback(Callback):
    def __init__(self, begin_unfreeze=5, max_epochs=20):
        super().__init__()
        self._begin_unfreezing = begin_unfreeze
        self._max_epochs = max_epochs
        self.epoch_counter = 0

    def on_train_start(self, trainer, pl_module) -> None:
        #orchestrate unfreezing
        clip = pl_module.transformer
        unfreeze_order = {self._begin_unfreezing: (clip.visual_projection, clip.text_projection),
                          self._max_epochs-1: (clip.vision_model.embeddings, clip.text_model.embeddings),
                         }
        n_text = len(clip.text_model.encoder.layers)
        n_vision = len(clip.vision_model.encoder.layers)
        start = self._begin_unfreezing+1
        end = self._max_epochs-1
        keys = list(range(start, end))
        n_keys = len(keys)

        for k in range(n_keys):
            text_layers = [layer for i, layer in enumerate(clip.text_model.encoder.layers) if k == np.floor(i*n_keys/n_text)]
            vision_layers = [layer for i, layer in enumerate(clip.vision_model.encoder.layers) if k == np.floor(i*n_keys/n_vision)]
            layers = text_layers+vision_layers
            if len(layers) > 0:
                unfreeze_order[end-1-k] = tuple(layers)
        
        self.unfreeze_order = unfreeze_order

    def on_train_epoch_end(self, trainer, pl_module):
        # When `current_epoch` is 10, feature_extractor will start training.
        self.epoch_counter+=1
        print(f'\n{self.epoch_counter} {self._begin_unfreezing}')
        if (self.epoch_counter >= self._begin_unfreezing) and (self.epoch_counter in self.unfreeze_order.keys()):
            print(f'\nUnfreezing modules:\n{self.unfreeze_order[self.epoch_counter]}')
            for layer in self.unfreeze_order[self.epoch_counter]:
                unfreeze(layer)
        

class PseudoLabellingCallback(Callback):
    def __init__(self, relabel_collate_fn, relabel_bs=128, confidence_th = .9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relabel_bs = relabel_bs
        self.confidence_th = confidence_th
        self.collate_fn = relabel_collate_fn

    def on_train_start(self, trainer, pl_module):
        self.original_dataset = trainer.train_dataloader.dataset

    def on_train_epoch_end(self, trainer, pl_module):
        print('Relabeling dataset:')
        temp_loader = DataLoader(self.original_dataset,
                                 batch_size=self.relabel_bs,
                                 shuffle=False,
                                 collate_fn=self.collate_fn,
                                 num_workers=8,
                                 prefetch_factor=4,
        )

        for encoded_batch, labels in tqdm(temp_loader):
            new_labels = []
            labels = labels.to(pl_module.device)
            encoded_batch = encoded_batch.to(pl_module.device)
            with torch.no_grad():
                x_hat_list = pl_module(**encoded_batch)
                preds = []
                
                for x_hat in x_hat_list:
                    high_conf = torch.where(x_hat.max(-1) > self.confidence_th, x_hat.argmax(-1), -100)
                    
                    preds.append(high_conf)

                pseudo_labels = torch.stack(preds, dim=-1)

                new_labels += torch.where(labels==-100, pseudo_labels, labels).cpu().numpy().tolist()

        trainer.train_dataloader.dataset = AugmentedWrapperDataset(self.original_dataset, new_labels)