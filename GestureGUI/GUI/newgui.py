import sys
import time
import cv2
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt, QStringListModel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit


class Communicate(QObject):
    closeApp = pyqtSignal()


class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changeText = pyqtSignal(QStringListModel)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.flagCyc = True
        self.flagVedio = False

    def run(self):
        # print('可以调用摄像头函数了')
        self.flagCyc = True
        self.imgNum = 0
        camera = cv2.VideoCapture(0)
        if camera is None:
            print('请先连接摄像头')
            exit()

        fps = 5  # 帧率
        pre_frame = None  # 总是取前一帧做为背景（不用考虑环境影响）

        count = 0
        # print('我在地方1死掉了')

        while self.flagCyc:
            # print('我在地方2死掉了')
            start = time.time()
            res, cur_frame = camera.read()
            if res != True: break
            end = time.time()
            seconds = end - start
            if seconds < 1.0 / fps:
                time.sleep(1.0 / fps - seconds)
            # print('读取了一帧')
            gray_img = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (500, 500))
            gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

            # 将图片显示到gui
            rgbImage = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

            # print(p)
            # QImage.scaled(self,rgbImage)

            self.changePixmap.emit(p)
            # self.changeText.emit(q)

            if pre_frame is None:
                pre_frame = gray_img
            else:
                # print('我在地方5死掉了')
                img_delta = cv2.absdiff(pre_frame, gray_img)
                thresh = cv2.threshold(img_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                flag = True
                for c in contours:
                    # 设置敏感度
                    if cv2.contourArea(c) > 2000:
                        print('动了')
                        flag = False
                        self.flagVedio = True
                        count = 0
                        continue
                    else:
                        break
                if self.flagVedio:
                    self.imgNum += 1
                    # 写入文件夹 还不OK
                    # cv2.imencode('.jpg', rgbImage)[1].tofile('images/')

                if flag and self.flagVedio:
                    count += 1
                if count == 10:
                    count = 0
                    print('调用神经网络')
                    self.flagVedio = False
                    # process_img('images/',self.imgNum)

                    # 还未完成，得到的文字emit到label
                pre_frame = gray_img


                # camera.release()
                # cv2.destroyAllWindows()

    def exit(self, returnCode=0):
        self.flagCyc = False


class GUI(QWidget):
    def __init__(self):
        super(GUI, self).__init__()
        self.initUi()

    def initUi(self):
        self.th = Thread(self)
        self.creatVboxGroupBox()
        self.creatHboxGroupBox()
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.hboxGroupBox)
        mainLayout.addWidget(self.vboxGroupBox)
        self.setLayout(mainLayout)

    def creatHboxGroupBox(self):
        self.hboxGroupBox = QGroupBox("Hbox layout")
        layout = QHBoxLayout()
        self.vedioLabel = QLabel(self)
        self.th.changePixmap.connect(self.vedioLabel.setPixmap)
        self.vedioLabel.setFixedWidth(640)
        self.vedioLabel.setFixedHeight(360)
        layout.addWidget(self.vedioLabel)
        self.hboxGroupBox.setLayout(layout)
        self.setWindowTitle('Basic Layout')

    def creatVboxGroupBox(self):
        self.vboxGroupBox = QGroupBox("Vbox layout")
        layout = QVBoxLayout()
        start = QPushButton('开始', self)
        # start.setCheckable(True)
        end = QPushButton('结束', self)
        # end.setCheckable(True)
        start.clicked.connect(lambda: self.cStart.closeApp.emit())
        end.clicked.connect(lambda: self.cEnd.closeApp.emit())
        self.text = QTextEdit()
        self.text.setPlainText("...")
        self.th.changeText.connect(self.text.setText)
        layout.addWidget(self.text)
        layout.addWidget(start)
        layout.addWidget(end)
        self.vboxGroupBox.setLayout(layout)
        # 信号和槽
        self.cStart = Communicate()
        self.cStart.closeApp.connect(lambda: self.th.start())
        self.th.wait()
        self.cEnd = Communicate()
        self.cEnd.closeApp.connect(lambda: self.th.exit())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    ex.show()
    sys.exit(app.exec_())
