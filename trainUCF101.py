#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: zz
@Date  : 2017/11/12
@Desc  :
    使用UCF101数据集训练模型
        随机选择10个分类训练，并根据7: 3分为训练集和测试及，50 EPOCHS --> 91.2% ACC

    1. 模型保存在target目录下，每10个epoch保存一次
    2. 定位到项目目录，执行 `tensorboard --logdir runs` 即可可视化训练过程，产生的log同样保存在上面
"""

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from datasets.GestureDataSet import GestureDataSet
from nets.C3dConvLstm import C3dConvLstmNet
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse

parser = argparse.ArgumentParser(description='UCF101 Training')
parser.add_argument('-d', '--data', default='/home/zdh/zz/workspace/refactorUCF101', type=str,
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size (default: 10)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seq-len', '--sl', default=16, type=int,
                    help='sequence length (default: 16)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    help='initial learning rate')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        # torch.nn.init.xavier_uniform(m.weight)
        m.weight.data.normal_(0.0, 0.02)
    # elif isinstance(m, nn.Linear):
    #     m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / 2


def evaluate(data_loader, writer, num_train_iters):
    net.eval()

    corrects = 0
    eval_num = 0
    for batch_idx, (data, target) in enumerate(data_loader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data, target = Variable(data), Variable(target)

        outputs = net(data)
        _, preds = torch.max(outputs.data, 1)

        eval_num += len(target)
        corrects += torch.sum(preds == target.data)

    accuracy = corrects / eval_num
    print('\nEvaluation -  acc: {:.3f}({}/{}) \n'.format(accuracy, corrects, eval_num))
    writer.add_text('Evaluation Log', 'Evaluation -  acc: {:.3f}%({}/{}) \n'.format(accuracy, corrects, eval_num),
                    num_train_iters)

    # 如果评估模型后需要再次训练，则标记
    # net.train()


writer = SummaryWriter()
args = parser.parse_args()


train_data = GestureDataSet(transform=transforms.Compose(
                        [transforms.Scale(112, interpolation=Image.CUBIC),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]), root=args.data, train=True, output_frames_cnt=args.seq_len)
test_data = GestureDataSet(transform=transforms.Compose(
                        [transforms.Scale(112, interpolation=Image.CUBIC),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]), root=args.data, train=False, output_frames_cnt=args.seq_len)

train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = data.DataLoader(dataset=test_data, batch_size=6, shuffle=True, num_workers=args.workers)

net = C3dConvLstmNet()
if torch.cuda.is_available():
    net.cuda()
net.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

num_train_iters = 0
for epoch in range(args.epochs):
    # Sets the module in training mode.
    # This has any effect only on modules such as Dropout or BatchNorm.
    net.train()

    if (epoch + 1) % 5 == 0:
        adjust_learning_rate(optimizer)

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        """
        data: (B, C, D, H, W)
        target: (Number, )
        """

        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, target)
        loss.backward()

        optimizer.step()

        num_train_iters += 1
        writer.add_scalar('loss', loss.data[0], num_train_iters)

        if (batch_idx + 1) % 5 == 0:
            _, preds = torch.max(outputs.data, 1)

            corrects = torch.sum(preds == target.data)
            losses = loss.data[0]
            accuracy = corrects / target.size(0)

            print('Train Epoch: [{}/{}], [{}/{} ({:.0f}%)] '
                  'Loss: {:.6f} BatchAcc: {:.3f} PredLabels: {} RealLabels: {}'.format(
                epoch + 1, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                accuracy,
                list(preds),
                list(target.data)))
            writer.add_text('Train Log', 'Train Epoch: [{}/{}], [{}/{} ({:.0f}%)] '
                            'Loss: {:.6f} BatchAcc: {:.3f}% PredLabels: {} RealLabels: {}'.format(
                            epoch + 1, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.data[0],
                            corrects / target.size(0),
                            list(preds),
                            list(target.data)), num_train_iters)

            writer.add_scalar('accuracy', accuracy, num_train_iters)

    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), 'target/C3dConvLstmNet_%s.pkl' % (epoch + 1))

    evaluate(test_loader, writer, num_train_iters)

# export scalar data to JSON for external processing
# writer.export_scalars_to_json("./all_scalars.json")

writer.close()