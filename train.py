import os
import time
import random
import argparse
import json
import numpy as np
import pandas as pd
import pdb
from pandas.io.parquet import FastParquetImpl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,f1_score
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='7')
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str)
    parser.add_argument('--data-dir', type=str, default='/home/ych/retinal_classfication/datasets/retinopathy')
    parser.add_argument('--out-dim', type=int, default=7)
    parser.add_argument('--image-size', type=int, default=224)  # resize后的图像大小
    parser.add_argument('--train-fold', type=str, default='0,1,2,3,4,5,6')
    parser.add_argument('--DEBUG', action='store_true', default=False)
    parser.add_argument('--freeze-cnn', action='store_true', default=False) # 冻结CNN参数
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--n-epochs', type=int, default=80)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--load-model', action='store_true', default=False) # 加载训练过的模型继续训练
    args, _ = parser.parse_known_args()
    return args


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)
    
def multi_label_f1(y_gt, y_pred, threshold=0.5):
    f1_out = []
    gt_np = y_gt.to("cpu").numpy()
    pred_np = (y_pred.to("cpu").numpy() > threshold) * 1.0
    assert gt_np.shape == pred_np.shape, "y_gt and y_pred should have the same size"
    for i in range(gt_np.shape[1]):
        f1_out.append(f1_score(gt_np[:, i], pred_np[:, i]))
    return f1_out


def val_epoch(model, loader, n_test=1, get_output=False):
    model.eval()
    val_loss = []
    out_pred = torch.FloatTensor().to(device)
    out_gt = torch.FloatTensor().to(device)  
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            out_gt = torch.cat((out_gt, target), 0)
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for I in range(n_test):
                l = model(get_trans(data, I))
                logits += l
            logits /= n_test
            # logits = model(data)
            out_pred = torch.cat((out_pred, logits), 0)
            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())
    val_loss = np.mean(val_loss)
    if get_output:
        return logits
    else:
        scores=np.array(multi_label_f1(out_gt, out_pred))
        return val_loss, scores.mean(), scores


def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        logits = model(data)

        loss = criterion(logits, target)
        loss.backward()

        # if args.image_size in [896, 576]:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  ??
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def run(folds, df, transforms_train, transforms_val):
    if args.DEBUG:
        args.n_epochs = 3
        df_train = df[df['fold'].isin(folds)].sample(args.batch_size * 4)
        df_valid = df[~df['fold'].isin(folds)].sample(args.batch_size * 4)
    else:
        df_train = df[df['fold'].isin(folds)]
        df_valid = df[~df['fold'].isin(folds)]

    dataset_train = RetinalDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = RetinalDataset(df_valid, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               sampler=RandomSampler(dataset_train),
                                               num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)
    model = ModelClass(
        args.enet_type,
        out_dim=args.out_dim,
        pretrained=True,
        freeze_cnn=args.freeze_cnn,
        load_model=args.load_model
    )
    para_num=sum(p.numel() for p in model.parameters() if p.requires_grad)
    content=f'Number of trainable parameters:{para_num}\n'
    if DP:
        pass
    model = model.to(device)

    score_max = 0.
    if args.DEBUG:
        model_file_best  = os.path.join(args.model_dir+'/debug', f'{args.kernel_type}_best.pth')
        model_file_final = os.path.join(args.model_dir+'/debug', f'{args.kernel_type}_final.pth')
    else:
        model_file_best  = os.path.join(args.model_dir, f'{args.kernel_type}_best.pth')
        model_file_final = os.path.join(args.model_dir, f'{args.kernel_type}_final.pth')
    
    if args.freeze_cnn:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if DP:
        pass        
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)
    nums = dataset_train.get_num()
    content+=f'total num of train:{len(dataset_train)},class nums:{nums}'+'\n'
    nums = dataset_valid.get_num()
    content+=f'total num of val:{len(dataset_valid)},class nums:{nums}'+'\n'
    if args.DEBUG:
        with open(os.path.join(args.log_dir+'/debug', f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content)
    else:
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content)
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Epoch {epoch}')

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, mean_score, scores= val_epoch(model, valid_loader)
        
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {val_loss:.5f}, mean_score: {mean_score:.4f}, scores: {scores[0]:.4f} {scores[1]:.4f} {scores[2]:.4f} {scores[3]:.4f} {scores[4]:.4f} {scores[5]:.4f} {scores[6]:.4f}.'
        print(content)
        if args.DEBUG:
            with open(os.path.join(args.log_dir+'/debug', f'log_{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')
        else:
            with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')
        scheduler_warmup.step()
        if epoch == 2: scheduler_warmup.step()  # bug workaround

        if mean_score > score_max:
            print('score_max ({:.6f} --> {:.6f}). Saving model ...'.format(score_max, mean_score))
            torch.save(model.state_dict(), model_file_best)
            score_max = mean_score

    torch.save(model.state_dict(), model_file_final)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    df,_ = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.train_fold.split(',')]
    run(folds, df, transforms_train, transforms_val)


if __name__ == '__main__':

    args = get_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir+'/debug', exist_ok=True)
    os.makedirs(args.log_dir+'/debug', exist_ok=True)
    if not args.kernel_type:
        args.kernel_type=f'{args.enet_type}_size{args.image_size}_outdim{args.out_dim}_bs{args.batch_size}'
        if args.freeze_cnn:
            args.kernel_type+='_freeze'

    if args.DEBUG:
        with open(os.path.join(args.log_dir+'/debug', f'log_{args.kernel_type}.txt'), 'w') as appender:
            args_str = json.dumps(vars(args), indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
            appender.write(args_str+"\n")
    else:
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'w') as appender:
            args_str = json.dumps(vars(args), indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
            appender.write(args_str+"\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed()

    device = torch.device('cuda')
    # LOSS WEIGHT
    pos_w = None
    w = None
    if False:
        pos_w=[1.3, 1.6, 4, 5.6, 5.2, 6, 5.6]
        w = [3., 1, 1, 1, 1,1, 1]
    
    criterion = nn.BCELoss(weight=w)

    main()