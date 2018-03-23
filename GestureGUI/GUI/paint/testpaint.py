from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QThread, QLineF
from PyQt5.QtWidgets import QWidget, QApplication
import sys
class Test(QThread):
    def __init__(self):
        super(Test, self).__init__()
        self.line = QLineF(10.0, 80.0, 90.0, 20.0)
        self.pa = QPainter()
        self.pa.drawLine(self.line)

if __name__ == '__main__':
    a = QApplication(sys.argv)
    app = Test()
    app.show()
    a.exec()
