#!/usr/bin/env python
# -*- coding:utf-8 -*-

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


class IsoGD(data.Dataset):
    def __init__(self, root, output_frames_cnt=32, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.output_frames_cnt = output_frames_cnt
        self.transform = transform
        self.target_transform = target_transform

        # 目前先仅处理RGB图像
        train_list_path = os.path.join(self.root, 'train_rgb_list.txt')

        assert os.path.exists(os.path.join(self.root, 'train')), "please make sure dir 'train' in %s" % self.root
        assert os.path.exists(
            os.path.join(self.root, train_list_path)), "please make sure file 'train_rgb_list.txt' in %s" % self.root

        self.train_list = None
        with open(train_list_path, 'r') as f:
            self.train_list = list(f.readlines())

    def __getitem__(self, item):
        video_info_list = self.train_list[item].split(' ')
        video_path = video_info_list[0]
        video_frame_cnt = int(video_info_list[1])
        target = video_info_list[2]

        seleted_frames = np.zeros(self.output_frames_cnt)
        scale = (video_frame_cnt - 1) * 1.0 / (self.output_frames_cnt - 1)
        if int(math.floor(scale)) == 0:
            pass
        else:
            seleted_frames[::] = np.floor(scale * np.arange(0, self.output_frames_cnt))

        # 如果不对图像进行预处理，将以图像默认size保存每一帧
        # 否则根据图像处理后结果保存
        sample_image_path = os.path.join(video_path, '%06d.jpg' % seleted_frames[0])
        assert os.path.exists(sample_image_path), "please make sure file '%s' exists" % sample_image_path
        sample_img = Image.open(sample_image_path)
        if self.transform is None:
            default_width, default_height = sample_img.size
            # res_images = np.empty((self.output_frames_cnt, default_height, default_width, 3), dtype=np.float32)
            res_images = torch.IntTensor(self.output_frames_cnt, default_height, default_width, 3).zero_()
        else:
            transform_img = self.transform(sample_img)  # torch.FloatTensor
            _, transform_height, transform_width = transform_img.size()
            res_images = torch.IntTensor(self.output_frames_cnt, transform_height, transform_width, 3).zero_()

        # 图像预处理并合并所有图像
        # 推荐处理：
        #   1. 随机截取最大正方形（这里240 * 240）
        #   1. resize -> 112 * 112
        #   2. 归一化 -> [-1.0, 1.0]
        for idx in range(0, self.output_frames_cnt):
            image_path = os.path.join(video_path, '%06d.jpg' % seleted_frames[idx])

            assert os.path.exists(image_path), "please make sure file '%s' exists" % image_path

            img = Image.open(image_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = torch.from_numpy(np.array(img))
            res_images[idx] = img

        if self.target_transform is not None:
            target = self.target_transform(target)
        return res_images, target

    def __len__(self):
        return len(self.train_list)