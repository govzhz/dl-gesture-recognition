#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: zz
@Date  : 2017/12/18
@Desc  :
    使用UCF101数据集训练的模型预测
"""

import torch
from nets.C3dConvLstm import C3dConvLstmNet
import argparse
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='UCF101 Predicting')
parser.add_argument('--model', default='/home/zdh/zz/workspace/AllNew/target/C3dConvLstmNet_30.pkl', type=str,
                    help='path to model')
parser.add_argument('--video-dir', default='/home/zdh/zz/workspace/refactorUCF101/test/rgb/v_BaseballPitch_g19_c04', type=str,
                    help='root path to video frames')


args = parser.parse_args()

transform_img = transforms.Compose(
        [transforms.Scale(112, interpolation=Image.CUBIC),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def process_img(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    square_sz = min(width, height)
    crop_h = int((height - square_sz) / 2)
    crop_w = int((width - square_sz) / 2)
    img = img.crop((crop_w, crop_h, square_sz + crop_w, square_sz + crop_h))
    return transform_img(img)


in_frame_cnt = len(os.listdir(args.video_dir))
output_frames_cnt = 16
seleted_frames = np.zeros(output_frames_cnt)
scale = (in_frame_cnt - 1) * 1.0 / (output_frames_cnt - 1)
if int(math.floor(scale)) == 0:
    seleted_frames[:in_frame_cnt] = np.arange(0, in_frame_cnt)
    seleted_frames[in_frame_cnt:] = in_frame_cnt - 1
else:
    seleted_frames[::] = np.floor(scale * np.arange(0, output_frames_cnt))

image_path = os.path.join(args.video_dir, '%06d.jpg' % seleted_frames[0])
img = process_img(image_path)
_, height, width = img.size()

data = torch.FloatTensor(3, output_frames_cnt, height, width).zero_()
data[:, 0, ...] = img

for idx in range(1, output_frames_cnt):
    image_path = os.path.join(args.video_dir, '%06d.jpg' % seleted_frames[idx])
    img = process_img(image_path)
    data[:, idx, ...] = img

data.unsqueeze_(0)
if torch.cuda.is_available():
    data = data.cuda()
data = Variable(data)

net = C3dConvLstmNet()
if torch.cuda.is_available():
    net.cuda()
net.load_state_dict(torch.load(args.model))

output = net(data)

_, pred = torch.max(output.data, 1)
pred = pred[0]

labelDict = {
    0: ('BandMarching', '行军演奏'),
    1: ('BasketballDunk', '扣篮'),
    2: ('BalanceBeam', '平衡木'),
    3: ('ApplyEyeMakeup', '眼部化妆'),
    4: ('Basketball', '打篮球'),
    5: ('Archery', '射箭'),
    6: ('ApplyLipstick', '抹口红'),
    7: ('BenchPress', '卧推'),
    8: ('BabyCrawling', '婴儿爬行'),
    9: ('BaseballPitch', '棒球投掷')
}

print('\nPrediction: %s, %s, %s' % (pred, labelDict[pred][0], labelDict[pred][1]))