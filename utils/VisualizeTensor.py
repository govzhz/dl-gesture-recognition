#!/usr/bin/env python
# encoding: utf-8

"""
@Author: zz
@Date  : 2017/11/12
@Desc  :
    可视化Tensor，确保图像预处理正确
"""

import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from datasets.GestureDataSet import GestureDataSet
import torch


def visualize_skig_tensor():
    """对用于训练的Skig数据集进行可视化

    """
    transform = transforms.Compose(
        [transforms.Scale(112, interpolation=Image.CUBIC),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = GestureDataSet(root='/home/zdh/zz/workspace/refactorSkig', train=True, output_frames_cnt=32,
                                transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)

    for batch_idx, (data, target) in enumerate(train_loader):
        images = torch.transpose(data[0, ...], 1, 0)  # (L, C, H, W)
        _imshow(torchvision.utils.make_grid(images), title='skig')


def visualize_isogd_tensor():
    """ 对用于训练的ISOGD数据集进行可视化

    """
    transform = transforms.Compose(
        [transforms.Scale(112, interpolation=Image.CUBIC),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = GestureDataSet(root='/home/zdh/zz/workspace/dataset', train=True, output_frames_cnt=32,
                                transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)

    """
    images, target = next(iter(train_loader))
    print(images[0, ...].size())  # (C, L, H, W)
    images = torch.transpose(images[0, ...], 1, 0)  # (L, C, H, W)
    # Make a grid from batch
    # Given a 4D mini-batch Tensor of shape (B x C x H x W),
    # or a list of images all of the same size,
    imshow(torchvision.utils.make_grid(images), title='gesture')
    """

    for batch_idx, (data, target) in enumerate(train_loader):
        images = torch.transpose(data[0, ...], 1, 0)  # (L, C, H, W)
        _imshow(torchvision.utils.make_grid(images), title='isogd')


def _imshow(inp, title=None):
    """Imshow for Tensor."""
    import matplotlib.pyplot as plt
    inp = inp.numpy().transpose((1, 2, 0))  # -> h, w, c
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


if __name__ == '__main__':
    # visualize_isogd_tensor()
    visualize_skig_tensor()