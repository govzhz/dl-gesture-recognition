#!/usr/bin/env python
# encoding: utf-8

"""
@Author: zz
@Date  : 2017/11/11
@Desc  :
    对IsoGD源数据集二次处理,用于生成可供训练的数据集
        1. createAll()：创建所有数据
        2. createTrainList()：创建训练集列表(分为depth和rgb)
        3. createValidList()：创建验证集列表(分为depth和rgb)
        4. createTrainDateSet(): 创建训练集
        5. createValidDateSet(): 创建验证集
    其中，classifyStartNum, classifyEndNum参数用于提取分类在 classifyStartNum <= classify < classifyEndNum
    范围内的数据作为训练集，默认提取全部分类

    Sample:
      > from utils.RefactoringSkig import IsoGD
      > isoGD = IsoGD('/home/zz/isogd_in', '/home/zz/isogd_out')  # 参数分别为输入数据集的根目录和输出数据集的根目录
      > isoGD.createAll()  # 提取全部构建数据集
      > isoGD.createAll(classifyStartNum=1, classifyEndNum=21)  # 提取分类为1-20构建数据集
"""

import os
from utils.BaseRefactoringDataset import BaseDataSet, ParseVideoException


class IsoGD(BaseDataSet):
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
        BaseDataSet.__init__(self, inRootDir, outRootDir)

        assert os.path.exists(os.path.join(inRootDir, 'train')), "please make sure dir 'train' in %s" % inRootDir
        assert os.path.exists(os.path.join(inRootDir, 'valid')), "please make sure dir 'valid' in %s" % inRootDir

        assert os.path.exists(
            os.path.join(inRootDir, 'train_list.txt')), "please make sure file 'train_list.txt' in %s" % inRootDir
        assert os.path.exists(
            os.path.join(inRootDir, 'valid_list.txt')), "please make sure file 'valid_list.txt' in %s" % inRootDir

    def createAll(self, classifyStartNum=None, classifyEndNum=None):
        """创建所有所需的文件
        """
        self._createSpecificList('train', classifyStartNum, classifyEndNum)
        self._createSpecificList('valid', classifyStartNum, classifyEndNum)
        self._createSpecificDataset('train', classifyStartNum, classifyEndNum)
        self._createSpecificDataset('valid', classifyStartNum, classifyEndNum)

    def createTrainList(self, classifyStartNum=None, classifyEndNum=None):
        """创建文件
         - train_rgb_list.txt
         - train_depth_list.txt
        """
        self._createSpecificList('train', classifyStartNum, classifyEndNum)

    def createValidList(self, classifyStartNum=None, classifyEndNum=None):
        """创建文件
         - valid_rgb_list.txt
         - valid_depth_list.txt
        """
        self._createSpecificList('train', classifyStartNum, classifyEndNum)

    def createTrainDateSet(self, classifyStartNum=None, classifyEndNum=None):
        """构建训练集"""
        self._createSpecificDataset('train', classifyStartNum, classifyEndNum)

    def createValidDateSet(self, classifyStartNum=None, classifyEndNum=None):
        """构建验证集"""
        self._createSpecificDataset('valid', classifyStartNum, classifyEndNum)

    def _createSpecificList(self, dataSetType, classifyStartNum=None, classifyEndNum=None):
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

                    inRgbVideoPath = os.path.join(self.inRootDir, tempList[0])
                    inDepthVideoPath = os.path.join(self.inRootDir, tempList[1])
                    classify = int(tempList[2])

                    if classifyStartNum is not None and classifyEndNum is not None:
                        assert classifyEndNum > classifyStartNum

                        if not classifyStartNum <= classify < classifyEndNum:
                            currentCount += 2
                            if currentCount % 200 == 0:
                                print("    process: %s / %s" % (currentCount, allCount))
                            continue
                    classify -= 1  # start with zero

                    outRgbVideoDir, rgbcount = self._getVideoInfo(dataSetType, inRgbVideoPath, 'rgb')
                    outDepthVideoDir, depthcount = self._getVideoInfo(dataSetType, inDepthVideoPath, 'depth')

                    frgb.write(outRgbVideoDir + ' ' + str(rgbcount) + ' ' + str(classify) + '\n')

                    currentCount += 1
                    if currentCount % 200 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))

                    fdepth.write(outDepthVideoDir + ' ' + str(depthcount) + ' ' + str(classify) + '\n')

                    currentCount += 1
                    if currentCount % 200 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))

                print('done\n')

    def _createSpecificDataset(self, dataSetType, classifyStartNum=None, classifyEndNum=None):
        print('create dataset: %s [%s]' % (dataSetType, os.path.join(self.outRootDir, dataSetType)))

        with open(os.path.join(self.inRootDir, dataSetType + '_list.txt'), 'r') as fread:
            f_lines = fread.readlines()
            allCount = len(f_lines) * 2

            currentCount = 0
            for line in f_lines:
                tempList = line.split(' ')

                if classifyStartNum is not None and classifyEndNum is not None:
                    assert len(tempList) == 3, 'current file not support, can not find out classify'
                    assert classifyEndNum > classifyStartNum

                    classify = int(tempList[2])
                    if not classifyStartNum <= classify < classifyEndNum:
                        currentCount += 2
                        if currentCount % 200 == 0:
                            print("    process: %s / %s" % (currentCount, allCount))
                        continue
                    classify -= 1  # start with zero

                inRgbVideoPath = os.path.join(self.inRootDir, tempList[0])
                inDepthVideoPath = os.path.join(self.inRootDir, tempList[1])

                outRgbVideoDir = self._getVideoInfo(dataSetType, inRgbVideoPath, 'rgb', onlyOutDir=True)
                outDepthVideoDir = self._getVideoInfo(dataSetType, inDepthVideoPath, 'depth', onlyOutDir=True)

                if not os.path.exists(outRgbVideoDir):
                    os.makedirs(outRgbVideoDir)
                if not os.path.exists(outDepthVideoDir):
                    os.makedirs(outDepthVideoDir)

                try:
                    self._toFrames(inRgbVideoPath, outRgbVideoDir)
                    currentCount += 1
                    if currentCount % 50 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))
                except ParseVideoException as e:
                    print(e)

                try:
                    self._toFrames(inDepthVideoPath, outDepthVideoDir)
                    currentCount += 1
                    if currentCount % 50 == 0:
                        print("    process: %s / %s" % (currentCount, allCount))
                except ParseVideoException as e:
                    print(e)

            print('done\n')


if __name__ == '__main__':
    isoGD = IsoGD('/home/zdh/zz/dataset/IsoGD_phase_1/IsoGD_phase_1', '/home/zdh/zz/workspace/qwe')
    isoGD.createAll(classifyStartNum=1, classifyEndNum=11)
