import sys
import cv2
import threading
import queue
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import json
import math
import io
import argparse

from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtWidgets import QWidget, QApplication, QTextEdit
from PyQt5.QtCore import pyqtSignal, QObject,QThread,QBasicTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5 import QtCore, QtGui, QtWidgets


labels = {
    "Doing other things": "正在做其他事（非手势）",
    "Drumming Fingers": "打击手指",
    "No gesture": "无手势",
    "Pulling Hand In": "拉近手",
    "Pushing Hand Away": "推远手",
    "Pulling Two Fingers In": "拉近两个手指",
    "Pushing Two Fingers Away": "推远两个手指",
    "Rolling Hand Backward": "",
    "Rolling Hand Forward": "",
    "Shaking Hand": "抖动手",
    "Sliding Two Fingers Down": "向下滑动两个手指",
    "Sliding Two Fingers Left": "向左滑动两个手指",
    "Sliding Two Fingers Right": "向右滑动两个手指",
    "Sliding Two Fingers Up": "向上滑动两个手指",
    "Stop Sign": "停止信号",
    "Swiping Down": "向下滑动",
    "Swiping Left": "向左滑动",
    "Swiping Right": "向右滑动",
    "Swiping Up": "向上滑动",
    "Thumb Down": "拇指向下",
    "Thumb Up": "竖起大拇指",
    "Turning Hand Clockwise": "顺时针转动手",
    "Turning Hand Counterclockwise": "逆时针转动手",
    "Zooming In With Full Hand": "整只手放大",
    "Zooming In With Two Fingers": "两个手指放大",
    "Zooming Out With Full Hand": "整只手缩小",
    "Zooming Out With Two Fingers": "两个手指缩小"
}

lists = ["Doing other things",
         "Drumming Fingers",
         "No gesture",
         "Pulling Hand In",
         "Pushing Hand Away",
         "Pulling Two Fingers In",
         "Pushing Two Fingers Away",
         "Rolling Hand Backward",
         "Rolling Hand Forward",
         "Shaking Hand",
         "Sliding Two Fingers Down",
         "Sliding Two Fingers Left",
         "Sliding Two Fingers Right",
         "Sliding Two Fingers Up",
         "Stop Sign",
         "Swiping Down",
         "Swiping Left",
         "Swiping Right",
         "Swiping Up",
         "Thumb Down",
         "Thumb Up",
         "Turning Hand Clockwise",
         "Turning Hand Counterclockwise",
         "Zooming In With Full Hand",
         "Zooming In With Two Fingers",
         "Zooming Out With Full Hand",
         "Zooming Out With Two Fingers"
         ]


parser = argparse.ArgumentParser(description='WebCam Network')
parser.add_argument('-s', '--server-address', default='http://127.0.0.1:5000', type=str,
                    help='手势识别服务器地址')
parser.add_argument('-d', '--device', default=0, type=int,
                    help='摄像头设备号')

# 图片上传大小（约27k）
UPLOAD_SIZE = (176, 100)
# 并发上传图片的线程池数
POOL_SIZE = 5


class Communicate(QObject):
    closeApp = pyqtSignal()
    closeApp2 = pyqtSignal(QTextEdit)

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changeProcessbar = pyqtSignal(tuple)
    changeText = pyqtSignal(str)
    def __init__(self, server_address, upload_size, pool_size, device_index, quit_key='q'):
        super(Thread, self).__init__()
        self._device_index = device_index  # 设备索引号或者视频
        self._quit = quit_key  # 退出摄像头
        self._server_address = server_address  # 手势识别服务器地址
        self._upload_size = upload_size  # 上传图片大小
        self._save = False  # 是否开始保存帧画面
        self._label = None  # 预测标签
        self._pro = None  # 预测准确率
        self._new = True  # 第一次打开系统标志
        self._queue_predict = queue.Queue()  # 保存预测结果
        self._pool = ThreadPoolExecutor(max_workers=pool_size)  # 线程池

        print("系统正在初始化...")
        self._cap_video = cv2.VideoCapture(device_index)  # 摄像头句柄
        self._save_distance = self.get_save_distance()  # 保存帧画面间距

    def get_save_distance(self):
        """计算保存帧画面间距"""
        # Number of frames to capture
        num_frames = 120

        print("Capturing {0} frames".format(num_frames))

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = self._cap_video.read()
            # cv2.imwrite(r"E:\df\%06d.jpg" % i, frame)

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))

        # Calculate frames per second
        fps = num_frames / seconds
        distance = math.ceil(fps / 5)
        print("Estimated frames per second : {0}".format(fps))
        print("Save distance: %s" % distance)
        return distance

        # 捕捉视频

    def run(self):
        frameCount = 0
        saveCount = 0
        flag = True
        flag2 = False
        while self._cap_video.isOpened():
            # 获取帧画面, 如果摄像头开启成功
            ret, frame = self._cap_video.read()
            frameCount += 1

            # 第一次加载系统提示
            if self._new:
                #frame = self._draw(frame, "系统准备完毕")
                self.changeText.emit("系统准备完毕")
                #self.blockSignals(True)
                self._new = False

            # 对帧画面操作
            if ret:
                if self._save and (frameCount + 1) % self._save_distance == 0:
                    self._pool.submit(self._upload_frame, saveCount, frame)
                    saveCount += 1


                # 读取预测结果
                if not self._queue_predict.empty():
                    try:
                        self._label, self._pro = self._queue_predict.get_nowait()
                        self.labelIndex = lists.index(self._label)
                        self.changeProcessbar.emit((self.labelIndex,self._pro))
                        self.changeText.emit(self._label)

                    except Exception as e:
                        print(e)

                # 显示预测结果
                if self._label is not None:
                    frame = self._draw(frame, labels[self._label])

                    # cv2.putText(frame, labels[self._label] + " - " + str(self._pro),
                    #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

                # 显示图像
                # cv2.imshow('Main', frame)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
                self.changePixmap.emit(convertToQtFormat)

            # 按q退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # 按s开始捕捉，再次按下结束捕捉同时开始预测
            if key == ord('s'):
                if not self._save:
                    # 初始化
                    self._label = None
                    self._pro = None
                    self._save = True
                    self._new = False
                    # 清空服务器数据
                    threading.Thread(target=self._remove).start()
                else:
                    self._save = False
                    # 预测分类
                    threading.Thread(target=self._predict).start()

        # 停止捕获视频
        self._cap_video.release()
        cv2.destroyAllWindows()

    def load_frames(self, inFrames, num_frames=8):
        in_frame_cnt = len(inFrames)

        if in_frame_cnt >= num_frames:
            seleted_frames = np.zeros(num_frames)
            scale = (in_frame_cnt - 1) * 1.0 / (num_frames - 1)
            if int(math.floor(scale)) == 0:
                seleted_frames[:in_frame_cnt] = np.arange(0, in_frame_cnt)
                seleted_frames[in_frame_cnt:] = in_frame_cnt - 1
            else:
                seleted_frames[::] = np.floor(scale * np.arange(0, num_frames))

            outFrames = [inFrames[index] for index in seleted_frames.astype(int)]
        else:
            raise ValueError('Video must have at least {} frames'.format(num_frames))

        return outFrames

    def _draw(self, frame, text):
        pil_im = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("font/simhei.ttf", 30, encoding='utf-8')
        draw.text((30, 30), text, (0, 0, 255), font=font)
        frame = np.array(pil_im)
        return frame

    def _predict(self):
        """预测分类"""
        start_time = time.time()
        res = json.loads(requests.get(self._server_address + "/category_network").text)
        if res["code"] == 0:
            print("Category ok...[%.4f]" % (time.time() - start_time))
            print(res["data"])
            res = json.loads(res["data"])
            category = max(res, key=res.get)
            self._queue_predict.put((category, res[category]))
        else:
            print("Category failed...[%.4f]" % (time.time() - start_time))
            self._queue_predict.put(("未知分类错误", 0))

    def _upload_frame(self, index, frame):
        """上传图片

        :param index: 图片索引
        :param frame: 帧画面
        :return: json字符串
        """
        start_time = time.time()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize(self._upload_size, Image.ANTIALIAS)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="PNG")
        imgByteArr = imgByteArr.getvalue()
        res = None
        try:
            res = requests.post(self._server_address + "/upload",
                                files={"image": ("%06d.jpg" % int(index), imgByteArr)}, timeout=(0.1, 0.8)).text
            print("Upload: %s[%.4f]" % (res, time.time() - start_time))
        except Exception as e:
            print(e)
        return res

    def _remove(self):
        """清空服务器图片数据

        :return: json字符串
        """
        start_time = time.time()
        res = json.loads(requests.get(self._server_address + "/remove").text)
        if res["code"] == 0:
            print("Remove ok...[%.4f]" % (time.time() - start_time))
        else:
            print("Remove failed...[%.4f]" % (time.time() - start_time))





class Ui_MainWindow(QWidget):

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.timer = QBasicTimer()
        self.step = 0
    def setupUi(self, MainWindow):
        args = parser.parse_args()
        self.th = Thread(self,upload_size=UPLOAD_SIZE, pool_size=POOL_SIZE,
                     device_index=args.device)
        self.graph1 = Communicate()
        self.font = QFont("Times",16);
        #self.font.setPointSizeF(10);
        self.font.setBold(True);
        #self.font.setPixelSize(10);
        #print(self.font.pointSize())
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(800, 490)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setAcceptDrops(False)
        MainWindow.setToolTipDuration(0)
        MainWindow.setStyleSheet("")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(500, 40, 51, 41))
        self.toolButton.setWhatsThis("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("resource/1200178.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon)
        self.toolButton.setObjectName("toolButton")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(490, 90, 81, 21))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.toolButton_2 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(690, 200, 51, 41))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resource/1200191.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_2.setIcon(icon1)
        self.toolButton_2.setObjectName("toolButton_2")
        self.toolButton_3 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_3.setGeometry(QtCore.QRect(690, 40, 51, 41))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resource/swipe_down_89.474060822898px_1200188_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_3.setIcon(icon2)
        self.toolButton_3.setObjectName("toolButton_3")
        self.toolButton_4 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_4.setGeometry(QtCore.QRect(600, 40, 51, 41))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("resource/like_77.762004175365px_1200177_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_4.setIcon(icon3)
        self.toolButton_4.setObjectName("toolButton_4")
        self.toolButton_5 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_5.setGeometry(QtCore.QRect(690, 120, 51, 41))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("resource/1200169.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_5.setIcon(icon4)
        self.toolButton_5.setObjectName("toolButton_5")
        self.toolButton_6 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_6.setGeometry(QtCore.QRect(600, 120, 51, 41))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("resource/zoom_in_98.436090225564px_1200197_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_6.setIcon(icon5)
        self.toolButton_6.setObjectName("toolButton_6")
        self.toolButton_7 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_7.setGeometry(QtCore.QRect(500, 120, 51, 41))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("resource/wave_99.585365853659px_1200195_easyicon.net.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_7.setIcon(icon6)
        self.toolButton_7.setObjectName("toolButton_7")
        self.toolButton_8 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_8.setGeometry(QtCore.QRect(500, 270, 51, 41))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("resource/1200173.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_8.setIcon(icon7)
        self.toolButton_8.setObjectName("toolButton_8")
        self.toolButton_9 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_9.setGeometry(QtCore.QRect(600, 200, 51, 41))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("resource/1200174.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_9.setIcon(icon8)
        self.toolButton_9.setObjectName("toolButton_9")
        self.toolButton_10 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_10.setGeometry(QtCore.QRect(500, 200, 51, 41))
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("resource/1200171.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_10.setIcon(icon9)
        self.toolButton_10.setObjectName("toolButton_10")
        self.toolButton_11 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_11.setGeometry(QtCore.QRect(600, 270, 51, 41))
        self.toolButton_11.setIcon(icon1)
        self.toolButton_11.setObjectName("toolButton_11")
        self.toolButton_12 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_12.setGeometry(QtCore.QRect(690, 270, 51, 41))
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("resource/1200184.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_12.setIcon(icon10)
        self.toolButton_12.setObjectName("toolButton_12")
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setGeometry(QtCore.QRect(580, 90, 81, 21))
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setGeometry(QtCore.QRect(680, 90, 81, 21))
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")
        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setGeometry(QtCore.QRect(680, 170, 81, 21))
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")
        self.progressBar_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_5.setGeometry(QtCore.QRect(490, 170, 81, 21))
        self.progressBar_5.setProperty("value", 0)
        self.progressBar_5.setObjectName("progressBar_5")
        self.progressBar_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_6.setGeometry(QtCore.QRect(580, 170, 81, 21))
        self.progressBar_6.setProperty("value", 0)
        self.progressBar_6.setObjectName("progressBar_6")
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setGeometry(QtCore.QRect(490, 250, 81, 21))
        self.progressBar_7.setProperty("value", 0)
        self.progressBar_7.setObjectName("progressBar_7")
        self.progressBar_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_8.setGeometry(QtCore.QRect(680, 250, 81, 21))
        self.progressBar_8.setProperty("value", 0)
        self.progressBar_8.setObjectName("progressBar_8")
        self.progressBar_9 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_9.setGeometry(QtCore.QRect(580, 250, 81, 21))
        self.progressBar_9.setProperty("value", 0)
        self.progressBar_9.setObjectName("progressBar_9")
        self.progressBar_10 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_10.setGeometry(QtCore.QRect(490, 320, 81, 21))
        self.progressBar_10.setProperty("value", 0)
        self.progressBar_10.setObjectName("progressBar_10")
        self.progressBar_11 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_11.setGeometry(QtCore.QRect(680, 320, 81, 21))
        self.progressBar_11.setProperty("value", 0)
        self.progressBar_11.setObjectName("progressBar_11")
        self.progressBar_12 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_12.setGeometry(QtCore.QRect(580, 320, 81, 21))
        self.progressBar_12.setProperty("value", 0)
        self.progressBar_12.setObjectName("progressBar_12")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(110, 360, 281, 51))
        self.textEdit.setStyleSheet("")
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 50, 361, 271))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(490, 370, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(630, 370, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.progressList = [self.progressBar,self.progressBar_2,self.progressBar_3,self.progressBar_4,self.progressBar_5,self.progressBar_6,
                        self.progressBar_7,self.progressBar_8,self.progressBar_9,self.progressBar_10,self.progressBar_11,self.progressBar_12]
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #连接
        self.th.changePixmap.connect(self.label.setPixmap)
        self.th.changeText.connect(self.textEdit.setText)
        #self.childTh.changeProcessbar.connect(self.updateProcessBar)
        self.th.changeProcessbar.connect(self.updateProcessBar)
        self.pushButton.clicked.connect(lambda: self.cStart.closeApp.emit())
        self.pushButton_2.clicked.connect(lambda: self.cEnd.closeApp.emit())
        self.cStart = Communicate()
        self.cStart.closeApp.connect(lambda: self.th.start())
        self.cEnd = Communicate()
        self.cEnd.closeApp.connect(lambda: self.th.stop())

        #还有好多个

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.toolButton_2.setText(_translate("MainWindow", "..."))
        self.toolButton_3.setText(_translate("MainWindow", "..."))
        self.toolButton_4.setText(_translate("MainWindow", "..."))
        self.toolButton_5.setText(_translate("MainWindow", "..."))
        self.toolButton_6.setText(_translate("MainWindow", "..."))
        self.toolButton_7.setText(_translate("MainWindow", "..."))
        self.toolButton_8.setText(_translate("MainWindow", "..."))
        self.toolButton_9.setText(_translate("MainWindow", "..."))
        self.toolButton_10.setText(_translate("MainWindow", "..."))
        self.toolButton_11.setText(_translate("MainWindow", "..."))
        self.toolButton_12.setText(_translate("MainWindow", "..."))
        self.textEdit.setText("hello")
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButton_2.setText(_translate("MainWindow", "End"))
        self.textEdit.setHtml(_translate("MainWindow",
                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'.SF NS Text\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
                                         "<p align=\"center\" dir=\'rtl\' style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt; color:#0080ff;\"></span></p></body></html>"))


    def updateProcessBar(self,val):
        step = 0
        num = val[1]
        classification = val[0]
        #print(self.font.pointSize())
        while step < num:
            step += 1
            self.progressList[classification].setValue(step)
            #self.progressList[classification].update(10,20,30,40)



    def setText(self,label):
        self.textEdit.setText(label)
        print(label)

    '''
        def timerEvent(self, event):
        print("2")
        if self.step >= 100:
            self.timer.stop()
            return
        self.step = self.step + 1
        self.progressBar.setValue(self.step)

    def onStart(self, val):
        num = val[1]
        classification = val[0]
        self.timer.start(100, self)
        #if self.timer.isActive():
            #self.timer.stop()
        #else:


    '''


    '''def setNumber(self,val,index):
        i = 0
        if i < val:
            i += 1
            self.progressList[index].setValue(i)
    '''


class MainUiClass(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(MainUiClass,self).__init__()
        self.setupUi(self)

if __name__ == '__main__':
    a = QApplication(sys.argv)
    app = MainUiClass()
    app.show()
    a.exec()
    args = parser.parse_args()
