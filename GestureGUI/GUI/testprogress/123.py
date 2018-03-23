# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '123.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
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

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QTextEdit

import  sys



class Communicate(QObject):
    closeApp = pyqtSignal()
    closeApp2 = pyqtSignal(QTextEdit)

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changeProcessbar = pyqtSignal(tuple)
    changeText = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()

    def run(self):
        self.changeText.emit("bbb")
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.th = Thread()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(150, 160, 261, 81))
        self.textEdit.setObjectName("textEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #self.textEdit.setText("aaa")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.th.changeText.connect(self.textEdit.setText)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


    def setText(self,label):
        self.textEdit.setText(label)

class MainUiClass(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,parent = None):
        super(MainUiClass,self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    app = MainUiClass()
    app.show()
    a.exec()