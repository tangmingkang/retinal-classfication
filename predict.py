import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import PIL.Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
from dataset import get_df, get_transforms, RetinalDataset
from models import Effnet, Resnest, Seresnext
from train import get_trans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str, default="")
    parser.add_argument('--data-dir', type=str, default='/home/tmk/project/retinal_classfication/datasets/')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--DEBUG', action='store_true', default=False)
    parser.add_argument('--val-fold', type=str, default='6,7,8,9')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--freeze-cnn', default=False)
    parser.add_argument('--eval', type=str, choices=['best', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')

    args, _ = parser.parse_known_args()
    return args


def main():

    df, _idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir
    )
    folds = [int(i) for i in args.val_fold.split(',')]
    df_test = df[df['fold'].isin(folds)]
    transforms_train, transforms_val = get_transforms(args.image_size)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = RetinalDataset(df_test, 'val', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)
    num_normal, num_hypertension = dataset_test.get_num()
    content=f'total num of test:{len(dataset_test)},normal examples:{num_normal},hypertension examples:{num_hypertension}'
    print(content)
    # load model
    models = []
    for fold in range(1):

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best.pth')
        elif args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final.pth')

        model = ModelClass(
            args.enet_type,
            out_dim=args.out_dim,
            pretrained=True,
            freeze_cnn=args.freeze_cnn
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        # if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        #     model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)

    # predict
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data,target) in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for model in models:
                for I in range(args.n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)

            probs /= args.n_test
            probs /= len(models)
            logits /= args.n_test
            logits /= len(models)

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    
    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    auc = roc_auc_score((TARGETS == _idx).astype(float), PROBS[:, _idx])
    acc_list_0=[]
    acc_list_1=[]
    for i in range(len(TARGETS)):
        if int(TARGETS[i])==0:
            acc_list_0.append(PROBS.argmax(1)[i]==TARGETS[i])
        elif int(TARGETS[i])==1:
            acc_list_1.append(PROBS.argmax(1)[i]==TARGETS[i])
    acc_0 = np.array(acc_list_0).mean() * 100.
    acc_1 = np.array(acc_list_1).mean() * 100.
    content = f'acc: {(acc):.4f}, auc: {(auc):.6f} acc_0: {(acc_0):.6f}, acc_1: {(acc_1):.6f}.'
    print(content)
    df_test['target'] = PROBS[:, _idx]
    df_test['evaluation']=(PROBS.argmax(1) == TARGETS).tolist()
    df_test[['image_name', 'target', 'diagnosis','evaluation']].to_csv(os.path.join(args.sub_dir, f'sub_{args.kernel_type}_{args.eval}.csv'), index=False)


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    if not args.kernel_type:
        args.kernel_type=f'{args.enet_type}_size{args.image_size}_outdim{args.out_dim}_bs{args.batch_size}'
        if args.freeze_cnn:
            args.kernel_type+='_freeze'
    if args.enet_type == 'resnest101':
        ModelClass = Resnest
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cuda')

    main()
