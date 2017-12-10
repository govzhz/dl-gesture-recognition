#!/usr/bin/env python
# encoding: utf-8

"""
@Author: zz
@Date  : 2017/12/10
@Desc  :
    功能：
        该类为对源数据集进行预处理,用于生成可供训练的数据集的基类，它定义了输出结构目录树，
        所有重构数据集的类必须实现该基类

    实现要求：
        1. 实现 createAll() 方法
        2. 视频输出路径不允许子类定义，需要从 _getVideoInfo() 方法中获取
        3. 为了方便阅读，请注明输入数据集和输出数据集的目录树
"""

import cv2
import os
import shutil


class BaseDataSet(object):

    def __init__(self, inRootDir, outRootDir):
        """输出目录树如下：
            - outRootDir
                - train
                   - rgb
                      - $[rgbVideoName1]
                         - 000000.jpg
                         - 000001.jpg
                         - ...
                      ...
                   - depth
                      - $[depthVideoName1]
                         - 000000.jpg
                         - 000001.jpg
                         - ...
                      ...
                   - train_rgb_list.txt
                   - train_depth_list.txt

                ...

        注：
           1. $[xxx]代表该名字是不确定的，它的值为每个视频的名字
           2. train_rgb_list.txt 每行内容为：每个视频帧画面输出根目录 视频帧数 分类（从0开始）
           3. 上述仅列出了 train 目录, valid 和 test 目录类似
           4. 生成目录不完全遵循上述结构，比如，当未传入 depth 视频，那么将不会生成相关文件

        :param inRootDir: 源数据集的根目录
        :param outRootDir: 输出可训练数据集的根目录
        """
        self.inRootDir = inRootDir
        self.outRootDir = outRootDir

    def createAll(self):
        """抽象方法：
            创建可供训练的所有训练集文件到输出目录
        """

        raise NotImplementedError("NotImplemented method `createAll`")

    def _getVideoInfo(self, dataSetType, invideoPath, videoType, onlyOutDir=False):
        """得到视频信息，包括如下：
             - 视频帧画面输出根目录
             - 视频帧画面数（可选）

        :param dataSetType: 所需创建数据集类型
                              - 'train': 训练集
                              - 'valid': 验证集
                              - 'test': 测试集
        :param invideoPath: 视频文件路径
        :param videoType: 视频类型
                            - 'rgb': rgb视频
                            - 'depth': depth视频
        :param onlyOutDir: 是否仅返回帧画面输出根目录
        :return: (帧画面输出根目录, 视频帧画面数) 或 帧画面输出根目录
        """
        outVideoDir = os.path.join(os.path.join(os.path.join(self.outRootDir, dataSetType),
                                                videoType), os.path.basename(invideoPath).split('.')[0])
        if not onlyOutDir:
            cap = cv2.VideoCapture(invideoPath)
            if not cap.isOpened():
                raise Exception('Could not open the video')
            count = self._countVideoFrames(cap)
            return outVideoDir, count
        return outVideoDir

    def _toFrames(self, inVideoPath, outVideoDir):
        """将视频文件解析为帧画面保存
            当解析帧数和实际视频帧数不同，会抛出ParseVideoException异常

        :param inVideoDir: 视频文件路径
        :param outVideoDir: 帧画面输出根目录
        """

        cap = cv2.VideoCapture(inVideoPath)
        if not cap.isOpened():
            raise Exception('Could not open the video')

        count = 0
        success = True
        while success:
            success, image = cap.read()
            if success:
                cv2.imwrite(os.path.join(outVideoDir, "%06d.jpg" % count), image)
                count += 1

        # 如果解析帧数和实际视频帧数不同，为了保险起见，
        # 将删除该视频所解析的帧画面，并抛出异常以便人工检查
        if count != self._countVideoFrames(cap):
            shutil.rmtree(outVideoDir)
            raise ParseVideoException("解析帧数和实际视频帧数不同")

    def _countVideoFrames(self, cap):
        """获取视频帧数

        :param cap: 视频流句柄
        :return: 视频帧数
        """
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


class ParseVideoException(Exception):
    def __init__(self, msg):
        self.message = msg

    def __str__(self):
        return self.message