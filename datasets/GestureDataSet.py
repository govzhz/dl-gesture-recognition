#!/usr/bin/env python
# encoding: utf-8

"""
@Author: zz
@Date  : 2017/11/12
@Desc  :
    IsoGD Dataset
"""


import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import os
import math
import random
from collections import namedtuple

Video = namedtuple('video', ['path', 'in_frame_cnt'])


def auto_crop(img, train):
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
    if train:
        crop_h = int((height - square_sz) * crop_random)
        crop_w = int((width - square_sz) * crop_random)
    else:
        crop_h = int((height - square_sz) / 2)
        crop_w = int((width - square_sz) / 2)
    return img.crop((crop_w, crop_h, square_sz + crop_w, square_sz + crop_h))


class GestureDataSet(data.Dataset):
    """支持遵循 BaseRefactoringDataset 输出目录结构（子类）的手语数据集

        TODO：
            1. 支持 depth 视频
            2. 支持 valid 数据集
    """

    def __init__(self, root, train, output_frames_cnt=32, transform=None, target_transform=None):
        assert transform is not None  # 因为没有transform，返回对象不是Tensor未进行处理，所以暂时不支持传入None

        self.root = os.path.expanduser(root)
        self.train = train
        self.output_frames_cnt = output_frames_cnt
        self.transform = transform
        self.target_transform = target_transform

        # 暂时不处理depth视频
        train_rgb_list_path = os.path.join(self.root, 'train_rgb_list.txt')
        test_rgb_list_path = os.path.join(self.root, 'test_rgb_list.txt')
        # 暂时不考虑验证集的处理，如果需要测试可以改为test使用
        # 这里只是占位，后期会加上对该数据集的支持
        valid_rgb_list_path = os.path.join(self.root, 'valid_rgb_list.txt')

        self.data_list = list()
        self.label_list = list()
        if train:
            assert os.path.exists(os.path.join(self.root, 'train')), "please make sure dir 'train' in %s" % self.root
            assert os.path.exists(os.path.join(self.root, 'train_rgb_list.txt')), \
                "please make sure file 'train_rgb_list.txt' in %s" % self.root
            self._parse_list_path(train_rgb_list_path)
        else:
            assert os.path.exists(os.path.join(self.root, 'test')), "please make sure dir 'test' in %s" % self.root
            assert os.path.exists(os.path.join(self.root, 'test_rgb_list.txt')), \
                "please make sure file 'test_rgb_list.txt' in %s" % self.root
            self._parse_list_path(test_rgb_list_path)

        assert len(self.data_list) == len(self.label_list)
        self.label_list = torch.LongTensor(self.label_list)

    def _parse_list_path(self, list_path):
        with open(list_path, 'r') as f:

            for line in f.readlines():
                video_info_list = line.split(' ')

                video_path = video_info_list[0]
                video_frame_cnt = int(video_info_list[1])
                target = int(video_info_list[2].replace('\n', ''))

                self.data_list.append(Video(video_path, video_frame_cnt))
                self.label_list.append(target)

    def __getitem__(self, item):
        """
        :return: (C, L, H, W)
        """
        path = self.data_list[item].path
        in_frame_cnt = self.data_list[item].in_frame_cnt
        target = self.label_list[item]

        seleted_frames = np.zeros(self.output_frames_cnt)
        scale = (in_frame_cnt - 1) * 1.0 / (self.output_frames_cnt - 1)
        if int(math.floor(scale)) == 0:
            seleted_frames[:in_frame_cnt] = np.arange(0, in_frame_cnt)
            seleted_frames[in_frame_cnt:] = in_frame_cnt - 1
        else:
            seleted_frames[::] = np.floor(scale * np.arange(0, self.output_frames_cnt))

        image_path = os.path.join(path, '%06d.jpg' % seleted_frames[0])
        img = self.process_img(image_path)
        _, height, width = img.size()

        imgs = torch.FloatTensor(3, self.output_frames_cnt, height, width).zero_()
        imgs[:, 0, ...] = img

        for idx in range(1, self.output_frames_cnt):
            image_path = os.path.join(path, '%06d.jpg' % seleted_frames[idx])
            img = self.process_img(image_path)
            imgs[:, idx, ...] = img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgs, target

    def __len__(self):
        return len(self.data_list)

    def process_img(self, image_path):
        """图像预处理

        1. 默认自动随机截取最大正方形（这里240 * 240）
        2. 需要transforms提供：
                a) resize -> 112 * 112
                b) 归一化 -> [-1.0, 1.0] # 应该预先获得数据集的均值
        :return: (C, H, W)
        """
        assert os.path.exists(image_path), "please make sure file '%s' exists" % image_path

        img = Image.open(image_path).convert('RGB')
        img = auto_crop(img, self.train)

        if self.transform is not None:
            img = self.transform(img)

        return img