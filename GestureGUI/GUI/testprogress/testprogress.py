from PyQt5.QtWidgets import QApplication, QProgressBar, QPushButton
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QBasicTimer
from PyQt5.QtWidgets import QLabel
from PyQt5 import QtGui
import time
class ProgressBar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)

        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('ProgressBar')
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)

        #toolButton
        self.toolButton = QtWidgets.QToolButton(self)
        self.toolButton.move(40, 120)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton.setIcon(icon)

        #toolButton2
        self.toolButton2 = QtWidgets.QToolButton(self)
        self.toolButton2.move(80, 120)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toolButton2.setIcon(icon)


        self.label = QLabel('aaa',self)
        self.label.move(40, 60)
        self.button = QPushButton('Start', self)
        self.button.setFocusPolicy(Qt.NoFocus)
        self.button.move(40, 80)
        self.button.clicked.connect(self.onStart)
        self.timer = QBasicTimer()
        self.step = 0

    def timerEvent(self, event):
        if self.step >= 100:
            self.timer.stop()
            return
        self.label.setStyleSheet("QLabel{border: 1px groove;border-radius:5px;border-style: outset;}");

        #self.toolButton.setStyleSheet("QToolButton{background:blue;}");
        #time.sleep(5)
        #self.toolButton2.setStyleSheet("QToolButton{background:blue;}");
        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def onStart(self):
        if self.timer.isActive():
            self.timer.stop()
            self.button.setText('Start')
        else:
            self.timer.start(100, self)
            self.button.setText('Stop')


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    qb = ProgressBar()
    qb.show()
    sys.exit(app.exec_())