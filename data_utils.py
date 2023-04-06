from dataset import Memotion7k, MultiOFF, FBHM, MMHS150K, BalancedInterleavedDataset

from transformers import AutoProcessor

import torch

from torch.utils.data import DataLoader

from torchvision.transforms import Compose

import torchvision.transforms as T

from niacin.augment import RandAugment
from niacin.text import en

DATASET_ZOO = {'memotion7k': Memotion7k,
               'multioff':MultiOFF,
               'fbhm':FBHM,
               'mmhs150k':MMHS150K,
               }

def text_augment(texts, augments):
    new_texts = []
    for text in texts:
        new_text = text
        for aug in augments:
            new_text = aug(new_text)
        new_texts.append(new_text)
    return new_texts


class MultimodalCollator:
    HARD_IMG_AUGMENTER = T.RandAugment(num_ops=6, magnitude=9)
    SOFT_IMG_AUGMENTER = Compose([T.RandomPerspective(.1, p=.5),
                                  T.RandomHorizontalFlip(p=.5),
                                ])
    HARD_TXT_AUGMENTER = RandAugment([en.add_synonyms,
                                      en.add_hyponyms,
                                      en.add_misspelling,
                                      en.add_contractions,
                                      en.add_whitespace,
                                      en.add_characters,
                                      en.add_fat_thumbs,
                                      en.remove_articles,
                                      en.remove_characters,
                                      en.remove_contractions,
                                      en.remove_punctuation,
                                      en.remove_whitespace,
                                      en.swap_chars,
                                      en.swap_words,
                                      en.add_bytes,
                                      en.add_leet,
                                    ], n=3, m=5, shuffle=True)
    SOFT_TXT_AUGMENTER = RandAugment([en.add_synonyms,
                                      en.add_whitespace,
                                      en.add_characters,
                                      en.remove_characters,
                                      en.remove_whitespace,
                                    ], n=1, m=3, shuffle=True)
    
    def __init__(self, processor, augment_mode='hard', split='train', max_length=40):
        # 40 max length for vilt // 77 max length for clip
        self.processor = AutoProcessor.from_pretrained(processor)
        self.split = split
        self.max_length = max_length
        self.augment_mode = augment_mode

    def __call__(self, batch):
        images, texts, labels = list(zip(*batch))
        if self.split=='train' and self.augment_mode == 'hard':
            images = [self.HARD_IMG_AUGMENTER(img) for img in images]
            texts = text_augment(texts, self.HARD_TXT_AUGMENTER)
        elif self.split=='train' and self.augment_mode == 'soft':
            images = [self.SOFT_IMG_AUGMENTER(img) for img in images]
            texts = text_augment(texts, self.SOFT_TXT_AUGMENTER)

        encoding = self.processor(images=images, 
                                  text=list(texts), 
                                  padding=True,
                                  max_length=self.max_length,
                                  truncation=True,
                                  return_tensors='pt')
        return encoding, torch.tensor(labels)

def data_loader(data, split, processor, bs=32, max_length=40):
    if data == 'all':
        dataset = BalancedInterleavedDataset([o for o in DATASET_ZOO.values()], split=split)
    else:    
        dataset = DATASET_ZOO[data](split)

    return DataLoader(dataset, 
                      batch_size=bs, 
                      shuffle= split=='train',
                      collate_fn=MultimodalCollator(processor=processor, 
                                                    split=split, 
                                                    max_length=max_length),
                      num_workers=8,
                      prefetch_factor=2,
                      pin_memory=True,
                      )

def class_weights(data):
    return DATASET_ZOO[data]('train').class_weights
