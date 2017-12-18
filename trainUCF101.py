#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: zz
@Date  : 2017/11/12
@Desc  :
    使用UCF101数据集训练模型
        随机选择10个分类训练，并根据7: 3分为训练集和测试及，50 EPOCHS --> 88.9% ACC
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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        # torch.nn.init.xavier_uniform(m.weight)
        m.weight.data.normal_(0.0, 0.02)
    # elif isinstance(m, nn.Linear):
    #     m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()

# def _initialize_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.01)
#             m.bias.data.zero_()

lr = 0.003
epochs = 50

writer = SummaryWriter()
transform = transforms.Compose(
    [transforms.Scale(112, interpolation=Image.CUBIC),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = GestureDataSet(root='/home/zdh/zz/workspace/refactorUCF101', train=True, output_frames_cnt=16, transform=transform)
test_data = GestureDataSet(root='/home/zdh/zz/workspace/refactorUCF101', train=False, output_frames_cnt=16, transform=transform)

train_loader = data.DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=4)
test_loader = data.DataLoader(dataset=test_data, batch_size=8, shuffle=True, num_workers=4)

net = C3dConvLstmNet()
if torch.cuda.is_available():
    net.cuda()
net.apply(weights_init)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


def eval_model(data_loader):
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
    print('\nEvaluation -  acc: {:.3f}%({}/{}) \n'.format(accuracy, corrects, eval_num))

    # 如果评估模型后需要再次训练，则标记
    # net.train()

num_iters = 0
for epoch in range(epochs):
    # Sets the module in training mode.
    # This has any effect only on modules such as Dropout or BatchNorm.
    net.train()

    if (epoch + 1) % 5 == 0:
        lr = lr / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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

        num_iters += 1
        writer.add_scalar('loss', loss.data[0], num_iters)  # data grouping by `slash`

        #print(net.group5[0].weight)
        optimizer.step()

        if (batch_idx + 1) % 4 == 0:
            _, preds = torch.max(outputs.data, 1)

            corrects = torch.sum(preds == target.data)
            losses = loss.data[0]

            print('Train Epoch: [{}/{}], [{}/{} ({:.0f}%)] '
                  'Loss: {:.6f} BatchAcc: {:.3f}% PredLabels: {} RealLabels: {}'.format(
                epoch + 1, epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],
                corrects / target.size(0),
                list(preds),
                list(target.data)))

    eval_model(test_loader)

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")

writer.close()