#!/usr/bin/env python
# encoding: utf-8

'''
@author: zz
@contact: zhzcoder@gmail.com
@time: 17-11-11 下午8:31
@desc:
    创建可供训练的数据集
        1. createAll()：创建所有数据
        2. createTrainList()：创建训练集列表(分为depth和rgb)
        3. createValidList()：创建验证集列表(分为depth和rgb)
        4. createTrainDateSet(): 创建训练集
        5. createValidDateSet(): 创建验证集

    Sample:
      > from IsoGDUtil import IsoGD
      > isoGD = IsoGD('/home/zz/dataset_in', '/home/zz/dataset_out')  # 参数分别为输入数据集的根目录和输出数据集的根目录
      > isoGD.createAll()
'''

import cv2
import os
import shutil


class IsoGD(object):
    def __init__(self, inRootDir, outRootDir):
        """
        IsoGD输入数据集目录树应如下：
         - inRootDir
            - train
               - 001
               - 002
                  - K_00001.avi
                  - M_00001.avi
                  - ...
               ...
            - valid
            - train_list.txt
            - valid_list.txt

        IsoGD输出数据集目录树如下：
         - outRootDir
            - train
               - rgb
                  - M_00001
                     - 000000.jpg
                     - 000001.jpg
                     - ...
                  ...
               - depth
                  - K_00001
                     - 000000.jpg
                     - 000001.jpg
                     - ...
                  ...
            - valid
            - train_rgb_list.txt
            - train_depth_list.txt
            - valid_rgb_list.txt
            - valid_depth_list.txt

        注意：
            1. IsoGD数据集的test分类给出的list没有视频分类，所以对模型的训练和验证没有帮助，所以不需要解析该分类
            2. IsoGD数据集的valid分类同样没有视频分类，但是目前有一份包含分类的list，在此list的基础上修改得到
        """
        assert os.path.exists(os.path.join(inRootDir, 'train')), "please make sure dir 'train' in %s" % inRootDir
        assert os.path.exists(os.path.join(inRootDir, 'valid')), "please make sure dir 'valid' in %s" % inRootDir

        assert os.path.exists(
            os.path.join(inRootDir, 'train_list.txt')), "please make sure file 'train_list.txt' in %s" % inRootDir
        assert os.path.exists(
            os.path.join(inRootDir, 'valid_list.txt')), "please make sure file 'valid_list.txt' in %s" % inRootDir

        self.inRootDir = inRootDir
        self.outRootDir = outRootDir

    def createAll(self):
        """创建所有所需的文件
        """
        self._createSpecificList('train')
        self._createSpecificList('valid')
        self._createSpecificDataset('train')
        self._createSpecificDataset('valid')

    def createTrainList(self):
        """创建文件
         - train_rgb_list.txt
         - train_depth_list.txt
        """
        self._createSpecificList('train')

    def createValidList(self):
        """创建文件
         - valid_rgb_list.txt
         - valid_depth_list.txt
        """
        self._createSpecificList('train')

    def createTrainDateSet(self):
        """构建训练集"""
        self._createSpecificDataset('train')

    def createValidDateSet(self):
        """构建验证集"""
        self._createSpecificDataset('valid')

    def _createSpecificList(self, dataSetType):
        print('create list: %s [%s]' % (dataSetType, self.outRootDir))
        rgbListFileName = os.path.join(self.outRootDir, dataSetType + '_rgb_list.txt')
        depthListFileName = os.path.join(self.outRootDir, dataSetType + '_depth_list.txt')

        assert not os.path.exists(rgbListFileName), rgbListFileName + ' exists'
        assert not os.path.exists(depthListFileName), depthListFileName + ' exists'

        with open(os.path.join(self.inRootDir, dataSetType + '_list.txt'), 'r') as fread, \
            open(rgbListFileName, 'w') as frgb, open(depthListFileName, 'w') as fdepth:
                inLines = fread.readlines()
                allCount = len(inLines) * 2

                currentCount = 0
                for inLine in inLines:
                    tempList = inLine.split(' ')

                    assert len(tempList) == 3, 'current file not support, can not find out classify'

                    rgbVideoDir = os.path.join(self.inRootDir, tempList[0])
                    depthVideoDir = os.path.join(self.inRootDir, tempList[1])
                    classify = int(tempList[2]) - 1  # start with zero

                    outRgbVideoDir, rgbcount = self._getVideoInfo(dataSetType, rgbVideoDir, 'rgb')
                    outDepthVideoDir, depthcount = self._getVideoInfo(dataSetType, depthVideoDir, 'depth')

                    frgb.write(outRgbVideoDir + ' ' + str(rgbcount) + ' ' + str(classify))

                    currentCount += 1
                    if currentCount % 200 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))

                    fdepth.write(outDepthVideoDir + ' ' + str(depthcount) + ' ' + str(classify))

                    currentCount += 1
                    if currentCount % 200 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))

                print('done\n')

    def _createSpecificDataset(self, dataSetType):
        print('create dataset: %s [%s]' % (dataSetType, os.path.join(self.outRootDir, dataSetType)))
        with open(os.path.join(self.inRootDir, dataSetType + '_list.txt'), 'r') as fread:
            f_lines = fread.readlines()
            allCount = len(f_lines) * 2

            currentCount = 0
            for line in f_lines:
                tempList = line.split(' ')

                inRgbVideoDir = os.path.join(self.inRootDir, tempList[0])
                inDepthVideoDir = os.path.join(self.inRootDir, tempList[1])

                outRgbVideoDir = self._getVideoInfo(dataSetType, inRgbVideoDir, 'rgb', onlyOutDir=True)
                outDepthVideoDir = self._getVideoInfo(dataSetType, inDepthVideoDir, 'depth', onlyOutDir=True)

                if not os.path.exists(outRgbVideoDir):
                    os.makedirs(outRgbVideoDir)
                if not os.path.exists(outDepthVideoDir):
                    os.makedirs(outDepthVideoDir)

                if self._toFrames(inRgbVideoDir, outRgbVideoDir):
                    currentCount += 1
                    if currentCount % 50 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))
                if self._toFrames(inDepthVideoDir, outDepthVideoDir):
                    currentCount += 1
                    if currentCount % 50 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))

            print('done\n')

    def _getVideoInfo(self, dataSetType, videoDir, videoType, onlyOutDir=False):
        outVideoDir = os.path.join(os.path.join(os.path.join(self.outRootDir, dataSetType),
                                                videoType), os.path.basename(videoDir).split('.')[0])
        if not onlyOutDir:
            cap = cv2.VideoCapture(videoDir)
            if not cap.isOpened():
                raise Exception('Could not open the video')
            count = self._countVideoFrames(cap)
            return outVideoDir, count
        return outVideoDir

    def _countVideoFrames(self, cap):
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _toFrames(self, inVideoDir, outVideoDir):
        cap = cv2.VideoCapture(inVideoDir)
        if not cap.isOpened():
            raise Exception('Could not open the video')

        count = 0
        success = True
        while success:
            success, image = cap.read()
            if success:
                cv2.imwrite(os.path.join(outVideoDir, "%06d.jpg" % count), image)
                count += 1

        if count != self._countVideoFrames(cap):
            shutil.rmtree(outVideoDir)
            print('[To frame failed] %s' % outVideoDir)
            return False
        return True

if __name__ == '__main__':
    isoGD = IsoGD('/home/zz/dataset_in', '/home/zz/dataset_out')
    isoGD.createAll()
