import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30, 30, 500, 300)

    def paintEvent(self, event):
        self.step=0
        painter = QPainter(self)
        path = QPainterPath()
        path.moveTo(20, 80);
        path.lineTo(20, 30);
        path.cubicTo(80, 0, 50, 50, 80, 80);
        painter.drawPath(path);
        painter.fillRect(20,30,1000,5,Qt.Dense1Pattern)
        painter.drawEllipse(50,60,13,9)
        '''
        pixmap = QPixmap("1200169.png")
        painter.drawPixmap(self.rect(), pixmap)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        while self.step < 50:
            self.step += 1
            painter.drawLine(10, 10, 10+self.step, 10)

        '''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())