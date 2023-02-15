import sys
sys.path.append('../')

from torch.utils import data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A

from glob import glob
import os
import os.path as osp
from glob import glob
import torch
import numpy as np
import cv2
import random
import pandas as pd
# import matplotlib.pyplot as plt

from utils.data_utils import augment
# from basicsr.utils import img2tensor
# from basicsr.utils.matlab_functions import rgb2ycbcr

"""
2022-02-16 @jzsherlock
return dict contains: 
    sar_img, eo_img, imid, class_id
"""

def stratified_sample_df(df, n_samples, random_seed):
    n = min(n_samples, df["class_id"].value_counts().min())
    df_ = df.groupby("class_id").apply(lambda x: x.sample(n, random_state=random_seed))
    df_.index = df_.index.droplevel(0)
    return df_, n

def stratified_trainval_split(df, n_samples, val_ratio):
    val_num = int(n_samples * val_ratio)
    df_val = df.groupby("class_id").apply(lambda x: x[: val_num])
    df_train = df.groupby("class_id").apply(lambda x: x[val_num: ])
    return df_train, df_val

def get_transformer(input_size):
    tfmr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])
    return tfmr

def strong_aug(p=0.5):
    return A.Compose([
        A.Resize(128, 128),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=p)

class SAREODataset(data.Dataset):

    def __init__(self, csv_path, dataroot, sar_size=128, eo_size=128, 
                phase='train', sample_per_class=0, val_ratio=0.1, random_seed=42, aug_mode='v0'):
        """
        sar + eo paired input dataset for classification
        """
        super(SAREODataset).__init__()

        self.phase = phase
        self.data_df = pd.read_csv(csv_path)
        self.dataroot = dataroot
        self.aug_mode = aug_mode

        if phase == "train" or phase == "valid":
            self.data_df, actual_sample_n = stratified_sample_df(self.data_df, sample_per_class, random_seed)
            print(f"[DATA] sampled {actual_sample_n} from each class in [train + val] set")
            if phase == 'train':
                self.data_df, _ = stratified_trainval_split(self.data_df, actual_sample_n, val_ratio=val_ratio)
            else:
                _, self.data_df = stratified_trainval_split(self.data_df, actual_sample_n, val_ratio=val_ratio)
        else:
            assert phase == 'test'
        
        self.dataset_len = len(self.data_df)
        print(f"[DATA] phase: {self.phase}, total dataset number: {self.dataset_len}")

        if self.aug_mode == 'v0':
            self.sar_tfmr = get_transformer(sar_size)
            self.eo_tfmr = get_transformer(eo_size)
        elif self.aug_mode == 'v1':
            self.auger = strong_aug(p=0.9)
            self.sar_tfmr = get_transformer(sar_size)
            self.eo_tfmr = get_transformer(eo_size)

    def __getitem__(self, index):
        
        imid = self.data_df.iloc[index]["image_id"]
        sar_path, eo_path = self.data_df.iloc[index][["sar_path", "eo_path"]]
        if self.phase == 'train' or self.phase == 'valid':
            class_id = self.data_df.iloc[index]["class_id"]
        img_sar = cv2.imread(osp.join(self.dataroot, sar_path))
        img_eo = cv2.imread(osp.join(self.dataroot, eo_path))

        if self.aug_mode == 'v0':
            img_sar, img_eo = augment([img_sar, img_eo], hflip=True, rotation=True)
            img_sar, img_eo = self.sar_tfmr(img_sar), self.eo_tfmr(img_eo)
        elif self.aug_mode == 'v1':
            img_sar = self.auger(image=img_sar)["image"]
            img_eo = self.auger(image=img_eo)["image"]
            img_sar, img_eo = self.sar_tfmr(img_sar), self.eo_tfmr(img_eo)

        if self.phase == 'train' or self.phase == 'valid':
            return {'img_sar': img_sar, 'img_eo': img_eo, 'imid': imid, 'class_id': class_id}
        else:
            return {'img_sar': img_sar, 'img_eo': img_eo, 'imid': imid}


    def __len__(self):
        return self.dataset_len


class PseudoSAREODataset(data.Dataset):

    def __init__(self, csv_path, dataroot, sar_size=128, eo_size=128, aug_mode='v0'):
        """
        sar + eo paired input dataset for classification
        """
        super(PseudoSAREODataset).__init__()

        self.data_df = pd.read_csv(csv_path)
        self.dataroot = dataroot
        self.aug_mode = aug_mode
        
        self.dataset_len = len(self.data_df)
        print(f"[DATA] total pseudo dataset number: {self.dataset_len}")

        if self.aug_mode == 'v0':
            self.sar_tfmr = get_transformer(sar_size)
            self.eo_tfmr = get_transformer(eo_size)
        elif self.aug_mode == 'v1':
            self.auger = strong_aug(p=0.9)
            self.sar_tfmr = get_transformer(sar_size)
            self.eo_tfmr = get_transformer(eo_size)


    def __getitem__(self, index):
        
        imid = self.data_df.iloc[index]["image_id"]
        sar_path, eo_path = self.data_df.iloc[index][["sar_path", "eo_path"]]
        class_id = self.data_df.iloc[index]["class_id"]
        img_sar = cv2.imread(osp.join(self.dataroot, sar_path))
        img_eo = cv2.imread(osp.join(self.dataroot, eo_path))

        if self.aug_mode == 'v0':
            img_sar, img_eo = augment([img_sar, img_eo], hflip=True, rotation=True)
            img_sar, img_eo = self.sar_tfmr(img_sar), self.eo_tfmr(img_eo)
        elif self.aug_mode == 'v1':
            img_sar = self.auger(image=img_sar)["image"]
            img_eo = self.auger(image=img_eo)["image"]
            img_sar, img_eo = self.sar_tfmr(img_sar), self.eo_tfmr(img_eo)

        return {'img_sar': img_sar, 'img_eo': img_eo, 'imid': imid, 'class_id': class_id}

    def __len__(self):
        return self.dataset_len


def get_trainval_sareo_dataloader(opt_dataset, random_seed):

    opt = opt_dataset['trainval']
    csv_path = opt['info_csv']
    dataroot = opt['dataroot']
    sample_per_class = opt['sample_per_class']
    val_ratio = opt['val_ratio']
    sar_size = opt['sar_input_size']
    eo_size = opt['eo_input_size']
    aug_mode = opt['aug_mode']

    batch_size = opt['batch_size']
    num_workers = opt['num_workers']

    train_dataset = SAREODataset(csv_path, dataroot, sar_size, eo_size, 'train', sample_per_class, val_ratio, random_seed, aug_mode)
    valid_dataset = SAREODataset(csv_path, dataroot, sar_size, eo_size, 'valid', sample_per_class, val_ratio, random_seed, aug_mode)

    train_imid_set = set(train_dataset.data_df["image_id"].values)
    valid_imid_set = set(valid_dataset.data_df["image_id"].values)
    print("train imid len, valid imid len ", len(train_imid_set), len(valid_imid_set))
    print("[train imid] intersection [valid imid] : (should be PHI) ", train_imid_set.intersection(valid_imid_set))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, \
                            pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=1, \
                            pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers)
    
    return train_loader, valid_loader

def get_pseudo_sareo_dataloader(opt_dataset):

    opt = opt_dataset['pseudo']
    csv_path = opt['info_csv']
    dataroot = opt['dataroot']
    sar_size = opt['sar_input_size']
    eo_size = opt['eo_input_size']
    aug_mode = opt['aug_mode']
    # batch_size = opt['batch_size']
    batch_size = 4
    num_workers = opt['num_workers']

    pseudo_dataset = PseudoSAREODataset(csv_path, dataroot, sar_size, eo_size, aug_mode)

    pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, \
                            pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers)
    
    return pseudo_loader


def get_test_sareo_dataloader(opt_dataset):
    
    opt = opt_dataset['test']
    csv_path = opt['info_csv']
    dataroot = opt['dataroot']
    sar_size = opt['sar_input_size']
    eo_size = opt['eo_input_size']

    test_dataset = SAREODataset(csv_path, dataroot, sar_size, eo_size, 'test')

    test_loader = DataLoader(test_dataset, batch_size=1, \
                            pin_memory=True, drop_last=False, shuffle=False, num_workers=0)
    
    return test_loader


if __name__ == "__main__":

    opt = {
        'trainval':{
            'type' :'SAREODataset',
            'info_csv': '/path/to/train_dataset_info.csv',
            'dataroot': '/path/to/dataset',
            'sample_per_class': 1000,
            'val_ratio': 0.1,
            'sar_input_size': 128,
            'eo_input_size': 128,
            'batch_size': 32,
            'num_workers': 8,
            'aug_mode': 'v1'
        },
        'test':{
            'type': 'SAREODataset',
            'info_csv': '/path/to/valid_dataset_info.csv',
            'dataroot': '/path/to/dataset',
            'sar_input_size': 128,
            'eo_input_size': 128
        }
    }

    train_loader, val_loader = get_trainval_sareo_dataloader(opt, random_seed=42)
    print(type(train_loader), type(val_loader))
    for iter_id, batch in enumerate(train_loader):
        if iter_id == 5:
            break
        print(iter_id, batch.keys())
        for k in batch.keys():
            print(f"{k} size: {batch[k].size()}, max: {torch.max(batch[k])}, min: {torch.min(batch[k])}")