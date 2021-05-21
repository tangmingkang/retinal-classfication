import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def shade_of_gray_cc(img, power=1, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    # print(img_power.shape, img_power[0][0][0])

    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    # print(rgb_vec)

    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))

    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)

def max_rgb(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    # print(img.shape)
    r_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    b_max = np.max(img[:, :, 2])
    max=np.max(img)

    # print(r_max,g_max,b_max,max)

    k=(r_max,g_max,b_max)/max
    res=img

    res[:, :, 0] = img[:, :, 0] / k[0]
    res[:, :, 1] = img[:, :, 1] / k[1]
    res[:, :, 2] = img[:, :, 2] / k[2]

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def max_red(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    # print(img.shape)
    b_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    r_max = np.max(img[:, :, 2])
    # max=np.max(img)

    # print(r_max,g_max,b_max,max)

    k=(b_max,g_max,r_max)/r_max
    res=img

    res[:, :, 0] = img[:, :, 0] / k[0]
    res[:, :, 1] = img[:, :, 1] / k[1]
    res[:, :, 2] = img[:, :, 2] / k[2]

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def QCGP(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    r_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    b_max = np.max(img[:, :, 2])
    k_max=(r_max+g_max+b_max)/3

    r_avg = np.mean(img[:, :, 0])
    g_avg = np.mean(img[:, :, 1])
    b_avg = np.mean(img[:, :, 2])
    k_avg = (r_avg + g_avg + b_avg) / 3

    r_u, r_v = cal(r_avg, r_max, k_avg, k_max)
    g_u, g_v = cal(g_avg, g_max, k_avg, k_max)
    b_u, b_v = cal(b_avg, b_max, k_avg, k_max)

    res = img

    res[:, :, 0] = (r_u *(img[:, :, 0] ** 2))+(r_v *(img[:, :, 0]))
    res[:, :, 1] = (g_u * (img[:, :, 1] ** 2)) + (g_v * (img[:, :, 1]))
    res[:, :, 2] = (b_u * (img[:, :, 2] ** 2)) + (b_v * (img[:, :, 2]))

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def cal(avg,max,k_avg,k_max):
    a = k_max-(k_avg*(max/avg))
    a=a/(max**2-avg*max)

    b=(k_avg/avg)-a*avg

    return a,b

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

        image = shade_of_gray_cc(image,power=1)
        # image = shade_of_gray_cc(image,power=6)
        # image = shade_of_gray_cc(image, power=12)
        # image = max_rgb(image)
        # image=max_red(image)
        # image = QCGP(image)



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
