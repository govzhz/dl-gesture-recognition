from utils.BaseRefactoringDataset import BaseDataSet, ParseVideoException
import os
from collections import defaultdict
import random
import math


class UCF101(BaseDataSet):

    def __init__(self, inRootDir, outRootDir):
        """
        UCF101输入数据集目录树应如下：
         - inRootDir
            - train
               - ApplyEyeMakeup
               - ApplyLipstick
               - BandMarching
                  - v_BandMarching_g01_c01.avi
                  ...

        Skig输出数据集目录树如下：
         - outRootDir
            - train
                - rgb
                    - v_BandMarching_g01_c01
                       - 000000.jpg
                       - 000001.jpg
                       - ...
                    ...
            - test
                - rgb
                    - v_BandMarching_g01_c06
                       - 000000.jpg
                       - 000001.jpg
                       - ...
                ...
            - train_list.txt
            - test_list.txt
        """
        BaseDataSet.__init__(self, inRootDir, outRootDir)

    def createAll(self):

        classifyDict = defaultdict(list)
        for classify, classifyName in enumerate(os.listdir(self.inRootDir)):

            classifyRootPath = os.path.join(self.inRootDir, classifyName)
            for videoName in os.listdir(classifyRootPath):
                classifyDict[classify].append((os.path.join(classifyRootPath, videoName), classify))

        trainList = list()
        testList = list()
        # classify start with zero
        for classify in classifyDict:
            # 随机打乱
            random.shuffle(classifyDict[classify])

            # test: train = 3: 7
            testCount = math.floor(len(classifyDict[classify]) * 3 / 10)
            testList += classifyDict[classify][:testCount]
            trainList += classifyDict[classify][testCount:]

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

            for videoDir, classify in dataSetList:
                outRgbVideoDir, countFrames = self._getVideoInfo(dataSetType, videoDir, 'rgb')

                fw.write(outRgbVideoDir + ' ' + str(countFrames) + ' ' + str(classify) + '\n')

                currentCount += 1
                if currentCount % 200 == 0:
                    print("    process: %s / %s" % (currentCount, allCount))

        print('done\n')

    def _createSpecificDataSet(self, dataSetType, dataSetList):
        print('create dataset: %s [%s]' % (dataSetType, os.path.join(self.outRootDir, dataSetType)))

        currentCount = 0
        allCount = len(dataSetList)
        for videoPath, _ in dataSetList:
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


if __name__ == '__main__':
    ucf101 = UCF101("/home/zdh/zz/dataset/UCF101/UCF-101", "/home/zdh/zz/workspace/refactorUCF101")
    ucf101.createAll()