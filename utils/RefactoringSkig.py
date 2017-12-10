#!/usr/bin/env python
# encoding: utf-8

"""
@Author: zz
@Date  : 2017/12/05
@Desc  :
    对Skig源数据集二次处理,用于生成可供训练的数据集
        createAll()：创建所有数据

    注意：多次调用createAll函数生成可训练文件时，请确保输出路径没有之前创建的 'train', 'test' 目录，
         因为在分配训练集和测试集过程中涉及随机打乱数据集，因此每次生成值并不相同！！！

    Sample:
        > from utils.RefactoringSkig import Skig
        > skig = Skig('/home/zz/skig_in', '/home/zz/skig_out')
        > skig.createAll()
"""

import os
from collections import defaultdict
import random
import math
from utils.BaseRefactoringDataset import BaseDataSet, ParseVideoException


class Skig(BaseDataSet):
    def __init__(self, inRootDir, outRootDir):
        """
        Skig输入数据集目录树应如下：
         - inRootDir
            - train
               - subject1_rgb
               - subject2_rgb
               - subject3_rgb
               - subject4_rgb
               - subject5_rgb
               - subject6_rgb
                  - M_person_6_backgroud_1_illumination_1_pose_1_actionType_1.avi
                  ...

        Skig输出数据集目录树如下：
         - outRootDir
            - train
                - rgb
                    - M_person_6_backgroud_1_illumination_1_pose_1_actionType_1
                       - 000000.jpg
                       - 000001.jpg
                       - ...
                    ...
            - test
                - rgb
                    - M_person_6_backgroud_1_illumination_1_pose_2_actionType_2
                       - 000000.jpg
                       - 000001.jpg
                       - ...
                ...
            - train_list.txt
            - test_list.txt
        """
        BaseDataSet.__init__(self, inRootDir, outRootDir)

        # 仅处理RGB视频，暂不处理depth视频
        for i in range(1, 7):
            assert os.path.exists(os.path.join(inRootDir, 'subject%s_rgb' % str(i))), \
                "please make sure dir 'subject%s_rgb' in %s" % (str(i), inRootDir)

        assert not os.path.exists(os.path.join(outRootDir, 'train')), "plear remove folder %s in %s" % ('train', outRootDir)
        assert not os.path.exists(os.path.join(outRootDir, 'test')), "plear remove folder %s in %s" % ('test', outRootDir)

    def createAll(self):
        """创建训练集以及测试集
            注意：Skig仅支持直接创建所有文件这一种方式，因为生成过程中随机打乱数据集用于分开测试集和训练集，
                每次随机值不同，所以需要保证txt中的路径正确对应视频路径
        """
        classifyDict = defaultdict(list)
        for i in range(1, 7):
            subjectDir = os.path.join(self.inRootDir, 'subject%s_rgb' % str(i))

            for videoName in os.listdir(subjectDir):
                key = videoName.split('_')[-1]
                classifyDict[key].append(os.path.join(subjectDir, videoName))

        trainList = list()
        testList = list()
        for key in classifyDict:
            # 随机打乱
            random.shuffle(classifyDict[key])

            # test: train = 3: 7
            allCount = len(classifyDict[key])
            testCount = math.floor(allCount * 3 / 10)
            testList += classifyDict[key][:testCount]
            trainList += classifyDict[key][testCount:]

        self._createSpecificList('train', trainList)
        self._createSpecificList('test', testList)
        self._createSpecificDataSet('train', trainList)
        self._createSpecificDataSet('test', testList)

    def _createSpecificList(self, dataSetType, dataSetList):
        print('create list: %s [%s]' % (dataSetType, self.outRootDir))

        listFileName = os.path.join(self.outRootDir, dataSetType + '_rgb_list.txt')
        with open(listFileName, 'w') as fw:
            currentCount = 0
            allCount = len(dataSetList)

            for videoDir in dataSetList:
                outRgbVideoDir, countFrames = self._getVideoInfo(dataSetType, videoDir, 'rgb')
                classify = int(os.path.basename(videoDir).split('.')[0].split('_')[-1])
                classify -= 1  # start with zero

                fw.write(outRgbVideoDir + ' ' + str(countFrames) + ' ' + str(classify) + '\n')

                currentCount += 1
                if currentCount % 200 == 0:
                    print("    process: %s / %s" % (currentCount, allCount))

        print('done\n')

    def _createSpecificDataSet(self, dataSetType, dataSetList):
        print('create dataset: %s [%s]' % (dataSetType, os.path.join(self.outRootDir, dataSetType)))

        currentCount = 0
        allCount = len(dataSetList)
        for videoPath in dataSetList:
            outVideoDir = self._getVideoInfo(dataSetType, videoPath, 'rgb', onlyOutDir=True)

            if not os.path.exists(outVideoDir):
                os.makedirs(outVideoDir)

            try:
                self._toFrames(videoPath, outVideoDir)
                currentCount += 1
                if currentCount % 50 == 0:
                    print("    process: %s / %s" % (currentCount, allCount))
            except ParseVideoException as e:
                print(e)

        print('done\n')


if __name__ == "__main__":
    skig = Skig('/home/zdh/zz/dataset/SKIG', '/home/zdh/zz/workspace/refactorSkig')
    skig.createAll()