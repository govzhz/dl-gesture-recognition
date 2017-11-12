#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: zz
@Date  : 2017/11/12
@Desc  :
    ...
"""

import random
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from datasets.IsoGD import IsoGD


def autoCrop(img):
    """自动随机裁剪最大正方形区域
        1. 训练集随机最大正方形区域
            crop_random = random.random()
            crop_h = int((height - square_sz) * crop_random)
            crop_w = int((width - square_sz) * crop_random)
        2. 测试集固定为整个图片的中心区域
            crop_h = int((image_h - square_sz) / 2)
            crop_w = int((image_w - square_sz) / 2)
    """
    width, height = img.size
    crop_random = random.random()
    square_sz = min(width, height)
    crop_h = int((height - square_sz) * crop_random)
    crop_w = int((width - square_sz) * crop_random)
    return img.crop((crop_w, crop_h, square_sz + crop_w, square_sz + crop_h))


transform = transforms.Compose(
    [transforms.Lambda(lambda img: autoCrop(img)),
     transforms.Scale(112, interpolation=Image.CUBIC),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_data = IsoGD(root='/home/zdh/zz/workspace/dataset', output_frames_cnt=32, transform=transform)

train_loader = data.DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=8)


for index, (data, target) in enumerate(train_loader):
    if torch.cuda.is_available():
        data = data.cuda()

    print(index, data.size(), target)





