import sys
import cv2
import argparse
import time
import math
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from ops.gesture_rec import GestureRec
from ops.limit_queue import LimitQueue
from darkflow.net.build import TFNet
import queue
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5 import QtCore, QtGui, QtWidgets

"""
基于目标跟踪的动态手势自动识别系统
"""

# 12分类
lists = [
    "抖动手指",
    "抖动手",
    "拉近两个手指",
    "两个手指放大",
    "两个手指缩小",
    "拇指向下",
    "竖起大拇指",
    "推远两个手指",
    "向下滑动两个手指",
    "向左滑动",
    "整只手放大",
    "整只手缩小",
    "非手势动作"  # 第13个分类！
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
# 自动识别判定总帧数（即通过n帧来判定）
DETECT_FRAMES_TOTAL = 5
# 动作开始所需帧数（不大于总帧数）
DETECT_FRAMES_START_ACTION = 4
# 动作结束所需帧数（不大于总帧数）
DETECT_FRAMES_END_ACTION = 4
# darkflow所需配置
DARKFLOW_OPTIONS = {
    "model": "cfg/yolo-gestures.cfg",
    "load": "model/yolo-gestures_15000.weights",
    "threshold": 0.1,
    'gpu': 0.5,
    "json": True
}


class Communicate(QObject):
    closeApp = pyqtSignal()


class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changePicture = pyqtSignal(QPixmap)
    changeProcessbar = pyqtSignal(tuple)
    changeText = pyqtSignal(str)
    changeBorder = pyqtSignal(int)
    changeBorderToNormal = pyqtSignal()

    def __init__(self):
        super(Thread, self).__init__()
        self._args = parser.parse_args()

        # 获取摄像头句柄
        self._cap_video = cv2.VideoCapture(self._args.device)
        # 显示结果队列
        self._queue_draw = queue.Queue()
        # 检测对象
        self._tfnet = list()
        # 初始化事件
        self._event = Event()
        # 开启线程池
        self._pool = ThreadPoolExecutor(max_workers=POOL_SIZE)
        # 子线程初始化
        self._frame_distance = list()
        self._pool.submit(self._get_upload_distance, self._event)
        self._queue_detect = LimitQueue(DETECT_FRAMES_TOTAL)  # 保存帧画面检测结果

        # 系统是否关闭的标志
        self.flag = True
        # 上传标志位
        self._is_upload = [None]

    def run(self):
        # 初始化阶段不执行识别逻辑
        if not self._event.isSet():
            self.changeText.emit("系统正在初始化，请稍等...")
            self._event.wait()

        # 获取识别对象
        self._gesture_rec = GestureRec(server_address=self._args.server_address, upload_size=UPLOAD_SIZE,
                                       frame_distance=self._frame_distance[0], queue_draw=self._queue_draw,
                                       pool=self._pool, tfnet=self._tfnet[0], queue_detect=self._queue_detect)

        frame_total = 0
        upload_total = 0
        if not self._cap_video.isOpened():
            self._cap_video = cv2.VideoCapture(self._args.device)
            self.changeText.emit("系统已重启")
        while self._cap_video.isOpened():
            if self.flag == 1:
                ret, frame = self._cap_video.read()
                frame_total += 1

                if ret:
                    self._gesture_rec.detection(frame_total, frame)

                    upload_total = self._gesture_rec.check_upload(frame_total, upload_total, frame, self._is_upload)

                    category_dict = self._gesture_rec.check_draw_text()
                    if category_dict is not None:
                        print(category_dict)

                        # 计算剩余进度条值
                        divided = self._processbar_generator(category_dict)
                        divided_index = 0

                        # 显示非预测分类结果
                        if divided is None:
                            self.changeText.emit(list(category_dict)[0])
                            continue

                        # 显示文本
                        show_category = max(category_dict, key=category_dict.get)
                        self.changeText.emit(show_category)
                        if show_category == "非手势动作":
                            for category in lists:
                                if category == "非手势动作":
                                    continue

                                self.changeProcessbar.emit((lists.index(category), 0))
                            continue
                        else:
                            # 显示分类图框框
                            self.changeBorder.emit(lists.index(show_category))

                        # 同时更新12个进度条
                        for category in lists:
                            if category == "非手势动作":
                                continue

                            if category in category_dict:
                                self.changeProcessbar.emit((lists.index(category), category_dict[category]))
                            else:
                                self.changeProcessbar.emit((lists.index(category), divided[divided_index]))
                                divided_index += 1

                    if self._is_upload[0] is None and self._queue_detect.getAliveSum() >= DETECT_FRAMES_START_ACTION:
                        # 动作开始，上传图片
                        self._is_upload[0] = True  # 标识识别开始
                        self.changeBorderToNormal.emit()
                        self.changeText.emit("请开始做动作")
                        self._gesture_rec.start_action()

                    if self._is_upload[0] and self._queue_detect.getAliveSum() <= \
                            (DETECT_FRAMES_TOTAL - DETECT_FRAMES_END_ACTION):
                        # 动作结束，预测
                        self._is_upload[0] = False  # 标识识别结束
                        self._gesture_rec.end_action(is_upload=self._is_upload)

                    self._show_frame(frame)
            else:
                # 初始化
                self.flag = True
                self._is_upload = [None]

                self._show_picture("resource/white.jpg")
                self.changeText.emit("系统停止")
                self._cap_video.release()

    def _processbar_generator(self, category_dict):
        """剩余进度条生成随机值
            算法：
                1. 构造一个N+1项的数组，第一项为0，最后一项为max
                2. 在[1, L)中随机选取N-1个不重复的正整数，并排序
                3. 所有的数组相邻两项的差值即为生成值
        """
        total = 100 - sum(category_dict.values())
        if total > 100:
            return None

        nums = 12 - len(category_dict)
        if "非手势动作" in category_dict:
            nums += 1

        # 当总值少于分配个数时，直接返回
        if total < nums:
            res = [0] * nums
            res[random.randint(0, nums - 1)] += total
            return res

        divided = []

        stick = [0] + random.sample(range(0, total), nums - 1) + [total]
        stick.sort()
        for i in range((len(stick) - 1)):
            divided.append((stick[i + 1] - stick[i]))
        return divided

    def _show_frame(self, frame):
        """帧画面显示于QT界面"""
        try:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            self.changePixmap.emit(convertToQtFormat)
        except Exception as e:
            pass

    def _show_picture(self, picture):
        try:
            img = cv2.imread(picture)
            rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            self.changePicture.emit(convertToQtFormat)
        except Exception as e:
            pass

    def _get_upload_distance(self, event):
        """计算保存帧画面间距"""
        # Number of frames to capture
        num_frames = 120
        print("系统初始化...")
        print("Capturing {0} frames".format(num_frames))

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = self._cap_video.read()

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

        self._tfnet.append(TFNet(DARKFLOW_OPTIONS))
        self._frame_distance.append(distance)
        self.changeText.emit("系统初始化完毕")

        event.set()

    def stop(self):
        self.flag = 0


class Ui_MainWindow(QWidget):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.step = 0

    def setupUi(self, MainWindow):
        args = parser.parse_args()
        self.th = Thread()
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1239, 755)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        MainWindow.setAcceptDrops(False)
        MainWindow.setToolTipDuration(0)
        MainWindow.setStyleSheet("QMainWindow{background:white;}")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.toolButton = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton.setGeometry(QtCore.QRect(820, 130, 81, 81))
        self.toolButton.setWhatsThis("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("resource/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon1)
        self.toolButton.setIconSize(QtCore.QSize(81, 81))
        self.toolButton.setObjectName("toolButton")
        self.toolButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_2 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_2.setGeometry(QtCore.QRect(970, 130, 81, 81))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("resource/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_2.setIcon(icon2)
        self.toolButton_2.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_2.setObjectName("toolButton_2")
        self.toolButton_2.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_3 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_3.setGeometry(QtCore.QRect(1110, 130, 81, 81))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("resource/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_3.setIcon(icon3)
        self.toolButton_3.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_3.setObjectName("toolButton_3")
        self.toolButton_3.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_4 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_4.setGeometry(QtCore.QRect(820, 240, 81, 81))
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("resource/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_4.setIcon(icon4)
        self.toolButton_4.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_4.setObjectName("toolButton_4")
        self.toolButton_4.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_5 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_5.setGeometry(QtCore.QRect(970, 240, 81, 81))
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("resource/5.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_5.setIcon(icon5)
        self.toolButton_5.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_5.setObjectName("toolButton_5")
        self.toolButton_5.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_6 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_6.setGeometry(QtCore.QRect(1110, 240, 81, 81))
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("resource/6.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_6.setIcon(icon6)
        self.toolButton_6.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_6.setObjectName("toolButton_6")
        self.toolButton_6.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_7 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_7.setGeometry(QtCore.QRect(820, 350, 81, 81))
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("resource/7.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_7.setIcon(icon7)
        self.toolButton_7.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_7.setObjectName("toolButton_7")
        self.toolButton_7.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_8 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_8.setGeometry(QtCore.QRect(970, 350, 81, 81))
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("resource/8.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_8.setIcon(icon8)
        self.toolButton_8.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_8.setObjectName("toolButton_8")
        self.toolButton_8.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_9 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_9.setGeometry(QtCore.QRect(1110, 350, 81, 81))
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("resource/9.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_9.setIcon(icon9)
        self.toolButton_9.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_9.setObjectName("toolButton_9")
        self.toolButton_9.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_10 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_10.setGeometry(QtCore.QRect(820, 460, 81, 81))
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("resource/10.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_10.setIcon(icon10)
        self.toolButton_10.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_10.setObjectName("toolButton_10")
        self.toolButton_10.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_11 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_11.setGeometry(QtCore.QRect(970, 460, 81, 81))
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap("resource/11.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_11.setIcon(icon11)
        self.toolButton_11.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_11.setObjectName("toolButton_11")
        self.toolButton_11.setFocusPolicy(QtCore.Qt.NoFocus)
        self.toolButton_12 = QtWidgets.QToolButton(self.centralwidget)
        self.toolButton_12.setGeometry(QtCore.QRect(1110, 460, 81, 81))
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap("resource/12.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton_12.setIcon(icon12)
        self.toolButton_12.setIconSize(QtCore.QSize(81, 81))
        self.toolButton_12.setObjectName("toolButton_12")
        self.toolButton_12.setFocusPolicy(QtCore.Qt.NoFocus)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(800, 210, 118, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setGeometry(QtCore.QRect(950, 210, 118, 23))
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setGeometry(QtCore.QRect(1090, 210, 118, 23))
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")
        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setGeometry(QtCore.QRect(800, 320, 118, 23))
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")
        self.progressBar_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_5.setGeometry(QtCore.QRect(950, 320, 118, 23))
        self.progressBar_5.setProperty("value", 0)
        self.progressBar_5.setObjectName("progressBar_5")
        self.progressBar_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_6.setGeometry(QtCore.QRect(1090, 320, 118, 23))
        self.progressBar_6.setProperty("value", 0)
        self.progressBar_6.setObjectName("progressBar_6")
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setGeometry(QtCore.QRect(800, 430, 118, 23))
        self.progressBar_7.setProperty("value", 0)
        self.progressBar_7.setObjectName("progressBar_7")
        self.progressBar_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_8.setGeometry(QtCore.QRect(950, 430, 118, 23))
        self.progressBar_8.setProperty("value", 0)
        self.progressBar_8.setObjectName("progressBar_8")
        self.progressBar_9 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_9.setGeometry(QtCore.QRect(1090, 430, 118, 23))
        self.progressBar_9.setProperty("value", 0)
        self.progressBar_9.setObjectName("progressBar_9")
        self.progressBar_10 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_10.setGeometry(QtCore.QRect(800, 540, 118, 23))
        self.progressBar_10.setProperty("value", 0)
        self.progressBar_10.setObjectName("progressBar_10")
        self.progressBar_11 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_11.setGeometry(QtCore.QRect(950, 540, 118, 23))
        self.progressBar_11.setProperty("value", 0)
        self.progressBar_11.setObjectName("progressBar_11")
        self.progressBar_12 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_12.setGeometry(QtCore.QRect(1090, 540, 118, 23))
        self.progressBar_12.setProperty("value", 0)
        self.progressBar_12.setObjectName("progressBar_12")
        # 文本框
        self.textEdit = QtWidgets.QLabel(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(180, 40, 400, 61))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setAlignment(Qt.AlignCenter)
        self.textEdit.setFont(QFont(QFont("Songti", 25, QFont.Normal)))
        # vedio label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 130, 661, 531))
        self.label.setObjectName("label")
        self.label.setScaledContents(True)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(820, 600, 160, 50))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1030, 600, 160, 50))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setFocusPolicy(QtCore.Qt.NoFocus)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.progressList = [self.progressBar, self.progressBar_2, self.progressBar_3, self.progressBar_4,
                             self.progressBar_5, self.progressBar_6, self.progressBar_7, self.progressBar_8,
                             self.progressBar_9, self.progressBar_10, self.progressBar_11, self.progressBar_12]
        self.retranslateUi(MainWindow)
        self.toolButtonList = [self.toolButton, self.toolButton_2, self.toolButton_3, self.toolButton_4,
                               self.toolButton_5, self.toolButton_6, self.toolButton_7, self.toolButton_8,
                               self.toolButton_9, self.toolButton_10, self.toolButton_11, self.toolButton_12]
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 连接
        self.th.changeBorderToNormal.connect(self.updateBorderToNormal)
        self.th.changeBorder.connect(self.updateBorder)
        self.th.changePicture.connect(self.label.setPixmap)
        self.th.changePixmap.connect(self.label.setPixmap)
        self.th.changeText.connect(self.textEdit.setText)
        self.th.changeProcessbar.connect(self.updateProcessBar)
        self.pushButton.clicked.connect(lambda: self.cStart.closeApp.emit())
        self.pushButton_2.clicked.connect(lambda: self.cEnd.closeApp.emit())
        self.cStart = Communicate()
        self.cStart.closeApp.connect(lambda: self.th.start())
        self.cEnd = Communicate()
        self.cEnd.closeApp.connect(lambda: self.th.stop())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手语识别系统"))
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
        self.pushButton.setText(_translate("MainWindow", "开始"))
        self.pushButton_2.setText(_translate("MainWindow", "结束"))
        self.textEdit.setText("系统初始化...")

    def updateProcessBar(self, val):
        self.progressList[val[0]].setValue(val[1])

    def updateBorder(self,val):
        self.toolButtonList[val].setStyleSheet("QToolButton{border: 5px blue;border-radius:15px;border-style: outset;}")

    def updateBorderToNormal(self):
        for i in range(12):
            self.toolButtonList[i].setStyleSheet("")

class MainUiClass(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainUiClass, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    app = MainUiClass()
    app.show()
    a.exec()
