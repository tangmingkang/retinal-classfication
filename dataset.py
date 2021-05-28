import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class RetinalDataset(Dataset):
    def __init__(self, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.num=0
        self.image_labels=[]
        for _, row in csv.iterrows():
            self.image_labels.append(row[1:-2])

    def __len__(self):
        return self.csv.shape[0]

    def get_num(self):
        names=['opacity','diabetic retinopathy','glaucoma','macular edema','macular degeneration','retinal vascular occlusion','normal']
        numbers=[]
        for name in names:
            numbers.append(self.csv.loc[(self.csv[name] == 1)].shape[0])
        return numbers
    
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath) # 默认读出的是BGR模式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # m*n*3
        
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32) # 512*512*3
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1) # 3*512*512
        data = torch.tensor(image).float()
        if self.mode == 'test':
            return data
        else:
            return data, torch.FloatTensor(self.image_labels[index])

def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

'''
df_train['image_name','image_path','diagnosis','fold']
'''
def get_df(kernel_type, out_dim, data_dir):
    df_train = pd.read_csv(os.path.join(data_dir, 'label.csv'), dtype={'filename':str})
    df_test = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'), dtype={'filename':str})
    df_train['filepath'] = df_train['filename'].apply(lambda x: os.path.join(data_dir, 'train/train/', x))
    df_test['filepath'] = df_test['filename'].apply(lambda x: os.path.join(data_dir, 'test/test/', x))
    return df_train,df_test
