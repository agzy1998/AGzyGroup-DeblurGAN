import sys

from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QMainWindow,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QFormLayout,QVBoxLayout,QLineEdit,QStackedLayout
from PyQt5.QtGui import QFont,QPalette
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from scripts.deblur_image2  import myDeblur


class DeblurTrain(QWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        vbo=QVBoxLayout()
        hbo=QHBoxLayout()
        hbo1=QHBoxLayout()
        stack=QStackedLayout()

        form1=QFormLayout()
        form2=QFormLayout()

        self.top1=QFrame()
        self.top2 = QFrame()
        self.topRight=QFrame()
        self.topLeft=QFrame()
        self.top=QFrame()
        self.bottom=QFrame()

        self.trainBtn=QPushButton('开\n始\n训\n练')

        self.trainBtn.setMaximumSize(50,175)
        self.trainBtn.setMinimumSize(50,175)
        stack.addWidget(self.trainBtn)
        self.topRight.setLayout(stack)

        imageNum=QLabel()
        imageNum.setText('图片数量')
        imageNum.setMinimumSize(100,26)
        imageNum.setMaximumSize(100,26)

        batchSize=QLabel()
        batchSize.setText('单次加载图片数量')
        batchSize.setMinimumSize(100, 26)
        batchSize.setMaximumSize(100, 26)

        logDir=QLabel()
        logDir.setText('输出路径')
        logDir.setMinimumSize(100, 26)
        logDir.setMaximumSize(100, 26)

        epochNum=QLabel()
        epochNum.setText('训练次数')
        epochNum.setMinimumSize(100, 26)
        epochNum.setMaximumSize(100, 26)

        criticUpdate=QLabel()
        criticUpdate.setText('鉴别次数')
        criticUpdate.setMinimumSize(100, 26)
        criticUpdate.setMaximumSize(100, 26)

        self.textEdit1=QLineEdit()
        self.textEdit1.setMaximumSize(200,50)
        self.textEdit1.setMinimumSize(200,10)

        self.textEdit2 = QLineEdit()
        self.textEdit2.setMaximumSize(200, 50)
        self.textEdit2.setMinimumSize(200, 10)

        self.textEdit3 = QLineEdit()
        self.textEdit3.setMaximumSize(200, 50)
        self.textEdit3.setMinimumSize(200, 10)

        self.textEdit4 = QLineEdit()
        self.textEdit4.setMaximumSize(200, 50)
        self.textEdit4.setMinimumSize(200, 10)

        self.textEdit5 = QLineEdit()
        self.textEdit5.setMaximumSize(200, 50)
        self.textEdit5.setMinimumSize(200, 10)




        form1.addWidget(imageNum)
        form1.addWidget(batchSize)
        form1.addWidget(logDir)
        form1.addWidget(epochNum)
        form1.addWidget(criticUpdate)

        form2.addWidget(self.textEdit1)
        form2.addWidget(self.textEdit2)
        form2.addWidget(self.textEdit3)
        form2.addWidget(self.textEdit4)
        form2.addWidget(self.textEdit5)

        self.top1.setLayout(form1)
        self.top2.setLayout(form2)

        hbo.addWidget(self.top1)
        hbo.addWidget(self.top2)

        self.topLeft.setLayout(hbo)

        hbo1.addWidget(self.topLeft)
        hbo1.addWidget(self.topRight)

        self.top.setLayout(hbo1)

        splitter=QSplitter(Qt.Vertical)
        splitter.addWidget(self.top)
        splitter.addWidget(self.bottom)
        splitter.setSizes([200,500])

        vbo.addWidget(splitter)
        #vbo.addWidget(self.top)
        #vbo.addWidget(self.bottom)

        self.setLayout(vbo)

        self.resize(800,800)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my = DeblurTrain()
    my.show()
    sys.exit(app.exec_())