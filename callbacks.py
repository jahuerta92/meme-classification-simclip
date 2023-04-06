
from pytorch_lightning.callbacks import Callback
from dataset import AugmentedWrapperDataset
from copy import deepcopy
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from tqdm.auto import tqdm

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