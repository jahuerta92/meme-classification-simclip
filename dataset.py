import json
from torch.utils.data import Dataset

from os.path import join, exists

from dicts import labels_memotion7k as LABELS_MEMOTION7K
from dicts import labels_multioff as LABELS_MULTIOFF

from sklearn.model_selection import train_test_split
from PIL import Image

from sklearn.utils import class_weight

import pandas as pd
import numpy as np
import re

from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DATA = 'local_data'


class DefaultDataset(Dataset):
    def __len__(self):
        return len(self.dataframe)

    def to_dataframe(self):
        return self.dataframe

class MMHS150K(DefaultDataset):
    FOLDER = join(ROOT_DATA, 'MMHS150K')
    DEV_METADATA = join(FOLDER, 'splits/val_ids.txt')
    TEST_METADATA = join(FOLDER, 'splits/test_ids.txt')
    TRAIN_METADATA = join(FOLDER, 'splits/train_ids.txt')
    GROUND_TRUTH = join(FOLDER, 'MMHS150K_GT.json')
    IMG_FOLDER = join(FOLDER, 'img_resized')
    TXT_FOLDER = join(FOLDER, 'img_txt')

    LABEL_SHAPE = (6,)
    TASK_TYPE = 'cls_prob'

    def __init__(self, split='train'):
        self.split = split
        gt = pd.read_json('local_data/MMHS150K/MMHS150K_GT.json', orient='index')
        gt.index = gt.index.astype(int).astype(str)

        if split == 'train':
            indices_file = self.TRAIN_METADATA
        elif split == 'dev':
            indices_file = self.DEV_METADATA
        elif split == 'test':
            indices_file = self.TEST_METADATA
        else:
            raise 'Only "train", "dev" or "test" allowed'
        
        with open(indices_file, 'r') as f:
            indices = f.readlines()
        
        all_idx = [i.strip() for i in indices]
        self.dataframe = gt.loc[all_idx]
        self.class_weights = [1, 1, 1, 1, 1, 1]

    def __getitem__(self, index):
        # RETURN FORMAT
        # image, text, labels
        row = self.dataframe.iloc[index]
        idx = self.dataframe.index[index]

        image = Image.open(join(self.IMG_FOLDER, f'{idx}.jpg')).convert('RGB')
        txt_path = join(self.TXT_FOLDER ,f'{idx}.json')
        text = str(row.tweet_text)

        if exists(txt_path):
            with open(txt_path) as f:
                text = text + '\n' + json.load(f)['img_text']

        text = re.sub(r"http\S+", "[URL]", text)
        labels = [0. for _ in range(self.LABEL_SHAPE[0])]
        for i in row.labels:
            labels[i] += 1.

        return image, text, labels
    
class FBHM(DefaultDataset):
    FOLDER = join(ROOT_DATA, 'FBHM')
    DEV_METADATA = join(FOLDER, 'dev.jsonl')
    TEST_METADATA = join(FOLDER, 'test.jsonl')
    TRAIN_METADATA = join(FOLDER, 'train.jsonl')
    LABEL_SHAPE = (1,)
    TASK_TYPE = 'cls'

    def __init__(self, split='train', text_option='text_corrected'):
        self.split = split

        if split == 'train':
            self.dataframe = pd.read_json(self.TRAIN_METADATA, lines=True)
        elif split == 'dev':
            self.dataframe = pd.read_json(self.DEV_METADATA, lines=True).sample(frac=1., random_state=19)
        elif split == 'test':
            self.dataframe = pd.read_json(self.TEST_METADATA, lines=True)
        else:
            raise 'Only "train", "dev" or "test" allowed'
        all_labels = self.dataframe.label.values
        self.class_weights = [class_weight.compute_class_weight(class_weight = 'balanced',
                                                               classes = np.unique(
                                                                   all_labels),
                                                               y = all_labels),]

    def __getitem__(self, index):
        # RETURN FORMAT
        # image, text, labels
        row = self.dataframe.iloc[index]
        img_file = row.img

        image = Image.open(join(self.FOLDER, img_file)).convert('RGB')
        text = str(row.text)
        label = int(row.label)

        return image, text, label


class MultiOFF(DefaultDataset):
    FOLDER = join(ROOT_DATA, 'MultiOFF')
    DEV_METADATA = join(FOLDER, 'Validation_meme_dataset.csv')
    TEST_METADATA = join(FOLDER, 'Testing_meme_dataset.csv')
    TRAIN_METADATA = join(FOLDER, 'Training_meme_dataset.csv')
    LABEL_2_IDX = LABELS_MULTIOFF
    LABEL_SHAPE = (1,)
    TASK_TYPE = 'cls'

    def __init__(self, split='train', text_option='text_corrected'):
        self.split = split

        if split == 'train':
            self.dataframe = pd.read_csv(self.TRAIN_METADATA)
        elif split == 'dev':
            self.dataframe = pd.read_csv(self.DEV_METADATA)
        elif split == 'test':
            self.dataframe = pd.read_csv(self.TEST_METADATA)
        else:
            raise 'Only "train", "dev" or "test" allowed'

        self.dataframe.label.replace(self.LABEL_2_IDX, inplace=True)

        all_labels = self.dataframe.label.values
        self.class_weights = [class_weight.compute_class_weight(class_weight = 'balanced',
                                                               classes= np.unique(
                                                                   all_labels),
                                                               y= all_labels),]

    def __getitem__(self, index):
        # RETURN FORMAT
        # image, text, labels
        row = self.dataframe.iloc[index]
        img_file = row.image_name

        image = Image.open(join(self.FOLDER, 'img', img_file)).convert('RGB')
        text = str(row.sentence)
        label = int(row.label)

        return image, text, label


class Memotion7k(DefaultDataset):
    FOLDER = join(ROOT_DATA, 'Memotion7k')
    TEST_FOLDER = join(FOLDER, 'memotion_test_dataset_7k')
    TRAIN_FOLDER = join(FOLDER, 'memotion_train_dataset_7k')
    DEV_P, DEV_R = .1, 42
    TEST_DATA = join(TEST_FOLDER, '2000_testdata.csv')
    TEST_GROUND_TRUTH = join(TEST_FOLDER, 'Meme_groundTruth.csv')

    LABEL_2_IDX = LABELS_MEMOTION7K
    TRAIN_DATA = join(TRAIN_FOLDER, 'labels.csv')
    TRAIN_IMG_FOLDER = join(TRAIN_FOLDER, 'images')
    TEST_IMG_FOLDER = join(TEST_FOLDER, '2000_data')

    LABEL_SHAPE = (4, 4, 4, 2, 3)
    TASK_TYPE = ['cls']*5

    def __init__(self, split='train', text_option='text_corrected'):
        self.split = split
        self.text_option = text_option

        if split == 'test':  # COLUMNS = image_name,text_ocr,text_corrected,humour,sarcasm,offensive,motivational,overall_sentiment
            data = pd.read_csv(self.TEST_DATA)
            data.columns = ['image_name', 'image_url',
                            'text_ocr', 'text_corrected']
            data.drop('image_url', axis=1, inplace=True)

            gt = pd.read_csv(self.TEST_GROUND_TRUTH)
            gt.columns = ['image_name', 'labels']

            joint_data = pd.merge(
                data, gt, left_on='image_name', right_on='image_name')

            humour, sarcasm, offensive, motivational, overall_sentiment, = [], [], [], [], []
            for item in joint_data.labels.tolist():
                sent, _, (hum, sarc, offe, motiv) = item.split('_')
                overall_sentiment.append(int(sent)+1)
                humour.append(int(hum))
                sarcasm.append(int(sarc))
                offensive.append(int(offe))
                motivational.append(int(motiv))

            joint_data['humour'] = humour
            joint_data['sarcasm'] = sarcasm
            joint_data['offensive'] = offensive
            joint_data['motivational'] = motivational
            joint_data['overall_sentiment'] = overall_sentiment

            self.dataframe = joint_data
            self.folder = self.TEST_IMG_FOLDER
        elif split == 'dev' or split == 'train':
            full_train = pd.read_csv(self.TRAIN_DATA)
            full_train.humour.replace(self.LABEL_2_IDX['humour'], inplace=True)
            full_train.sarcasm.replace(
                self.LABEL_2_IDX['sarcasm'], inplace=True)
            full_train.offensive.replace(
                self.LABEL_2_IDX['offensive'], inplace=True)
            full_train.motivational.replace(
                self.LABEL_2_IDX['motivational'], inplace=True)
            full_train.overall_sentiment.replace(
                self.LABEL_2_IDX['overall_sentiment'], inplace=True)

            train, dev = train_test_split(
                full_train, test_size=self.DEV_P, random_state=self.DEV_R)
            if split == 'dev':
                self.dataframe = dev
            else:
                self.dataframe = train
            self.folder = self.TRAIN_IMG_FOLDER

        else:
            raise 'Only "train", "dev" or "test" allowed'

        all_labels = [self.dataframe.humour.values,
                      self.dataframe.sarcasm.values,
                      self.dataframe.offensive.values,
                      self.dataframe.motivational.values,
                      self.dataframe.overall_sentiment.values]

        self.class_weights = [class_weight.compute_class_weight(
                                class_weight = 'balanced', 
                                classes = np.unique(label), 
                                y = label) for label in all_labels]

    def __getitem__(self, index):
        # RETURN FORMAT
        # image, text, labels
        row = self.dataframe.iloc[index]
        img_file = row.image_name

        image = Image.open(join(self.folder, img_file)).convert('RGB')
        text = str(row[self.text_option])
        labels = [row.humour, row.sarcasm, row.offensive,
                  row.motivational, row.overall_sentiment]
        return image, text, labels

class BalancedInterleavedDataset(Dataset):
    def __init__(self, dataset_objects, split='train', text_option='text_corrected'):
        self.datasets = [obj(split=split, text_option=text_option) for obj in dataset_objects]
        self.LABEL_SHAPE = [l for dataset in self.datasets for l in dataset.LABEL_SHAPE]
        self.class_weights = [cw for dataset in self.datasets for cw in dataset.class_weights]

        self.ind_label_len = [len(dataset.LABEL_SHAPE) for dataset in self.datasets]
        self.split = split

        self.n = len(self.datasets)
        self.max_length = max(len(dataset) for dataset in self.datasets)
    
    def __len__(self):
        return self.max_length * self.n
    
    def __getitem__(self, index): 
        # RETURN FORMAT
        # image, text, labels
        id_data = index%self.n
        id_row = index//self.n

        sub_data = self.datasets[id_data]
        image, text, begin_labels = sub_data[id_row%len(sub_data)]

        labels = [-100 for _ in range(len(self.LABEL_SHAPE))]
        begin_label_idx = sum(self.ind_label_len[:id_data])

        if type(begin_labels) is list:
            for i, l in enumerate(begin_labels):
                labels[i+begin_label_idx] = l
        else:
            labels[begin_label_idx] = begin_labels

        return image, text, labels

class AugmentedWrapperDataset(Dataset):
    def __init__(self, old_dataset, new_labels):
        self.dataset = old_dataset
        self.labels = new_labels
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, text, _ = self.dataset[index] 
        labels = self.labels[index]
        return image, text, labels      
