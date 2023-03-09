from dataset import Memotion7k, MultiOFF, FBHM

from transformers import AutoProcessor

import torch

from torch.utils.data import DataLoader

from torchvision.transforms import RandAugment

DATASET_ZOO = {'memotion7k': Memotion7k,
               'multioff':MultiOFF,
               'fbhm':FBHM}

class MultimodalCollator:
    def __init__(self, processor, split='train'):
        self.processor = AutoProcessor.from_pretrained(processor)
        self.split = split
        self.augmenter = RandAugment(num_ops=7)
    
    def __call__(self, batch):
        images, texts, labels = list(zip(*batch))
        if self.split=='train':
            images = [self.augmenter(img) for img in images]
        encoding = self.processor(images, list(texts), 
                                  padding='max_length',
                                  max_length=40,
                                  truncation=True,
                                  return_tensors='pt')
        return encoding, torch.tensor(labels)

def data_loader(data, split, processor, bs=32):
    dataset = DATASET_ZOO[data](split)
    return DataLoader(dataset, 
                      batch_size=bs, 
                      shuffle= split=='train',
                      collate_fn=MultimodalCollator(processor),
                      num_workers=8,
                      prefetch_factor=2,
                      pin_memory=True
                      )

def class_weights(data):
    return DATASET_ZOO[data]('train').class_weights
