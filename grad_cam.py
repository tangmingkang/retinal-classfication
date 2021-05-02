# -*- coding: utf-8 -*-
import argparse
import os
import re
import albumentations
import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models
from grad_cam.interpretability.grad_cam import GradCAM, GradCamPlusPlus
from grad_cam.interpretability.guided_back_propagation import GuidedBackPropagation
from models import Effnet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='effnet')
    parser.add_argument('--enet-type', type=str, default='efficientnet_b3')
    parser.add_argument('--kernel-type', type=str)
    parser.add_argument('--out-dim', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=512)  # resize后的图像大小
    parser.add_argument('--freeze-cnn', default=False)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval', type=str, choices=['best', 'final'], default="best")
    parser.add_argument('--root-path', type=str, default='/home/tmk/project/retinal_classfication/',
                        help='input image path')
    parser.add_argument('--weight-path', type=str, default='weights/')
    parser.add_argument('--image-path', type=str, default='datasets/image/000326720200314010002.jpg',
                        help='input image name')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--class-id', type=int, default=None,
                        help='class id') # Grad-CAM和Guided Back Propagation反向传播使用的类别id（可选,默认网络预测的类别)
    parser.add_argument('--output-dir', type=str, default='visualization/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    elif net_name in ['effnet']:
        net = Effnet(
        args.enet_type,
        out_dim=args.out_dim,
        pretrained=True,
        freeze_cnn=args.freeze_cnn
    )

    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
    elif weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name




def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        os.makedirs(output_dir+str(key),exist_ok=True)
        io.imsave(output_dir+str(key)+f'/{prefix}.jpg',image)

def main():
    transforms = albumentations.Compose([
        albumentations.Resize(args.image_size, args.image_size),
        albumentations.Normalize()
    ])
    image = cv2.imread(args.root_path+args.image_path) # 默认读出的是BGR模式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # m*n*3
    res = transforms(image=image)
    image = res['image'].astype(np.float32)
    image = image.transpose(2, 0, 1) # 3*512*512
    image = image[np.newaxis, ...]  # 增加batch维
    inputs = torch.tensor(image, requires_grad=True)

    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (512, 512))) / 255

    # 输出图像
    image_dict = {}
    # 网络
    net = get_net(args.network, args.root_path+args.weight_path+args.kernel_type)
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(inputs, args.class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.root_path+args.output_dir)


if __name__ == '__main__':
    args=get_args()
    os.makedirs(args.root_path+args.output_dir, exist_ok=True)
    if not args.kernel_type:
        args.kernel_type=f'{args.enet_type}_size{args.image_size}_outdim{args.out_dim}_bs{args.batch_size}'
        if args.freeze_cnn:
            args.kernel_type+='_freeze'
        args.kernel_type+=f'_{args.eval}.pth'
    else:
        args.kernel_type+=f'_{args.eval}.pth'
    main()
