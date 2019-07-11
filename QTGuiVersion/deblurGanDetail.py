import sys
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QMainWindow,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QGraphicsOpacityEffect,QScrollBar,QTextEdit,QScrollArea
from PyQt5.QtGui import QFont,QPalette
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

class detail(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.scroll_area = QScrollArea()
        self.scroll_contents = QStackedWidget()
        self.scroll_contents.setMinimumSize(800,1700)
        self.scroll_contents.setStyleSheet('border:none;')
        self.scroll_area.setStyleSheet('boder:none;')

        label = QLabel(self.scroll_contents)
        label.setText('Deblugan your image')
        label.move(150, 10)
        label.resize(420, 30)
        label.setStyleSheet(" border:4px;border-color:black;color:red")
        label.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))

        label1=QLabel(self.scroll_contents)
        label1.setText('一、背景介绍')
        label1.move(10,60)
        label1.resize(150,20)
        label1.setStyleSheet(" border:4px;border-color:black;color:red")
        label1.setFont(QFont("Microsoft YaHei",13))

        text1=QTextEdit(self.scroll_contents)
        text1.setText(" 置身异国街道，感受着陌生环境里熙熙攘攘的街道，你掏出手机想留住这一刻。好嘞，一、二、三，咔嚓。由于行人和车辆都在运动，再加上你的手稍微抖了一抖，照片中的景象是这样的——")
        text1.move(20,90)
        text1.resize(700,60)
        text1.setStyleSheet('background-color:white')
        text1.setReadOnly(True)

        labelImg1=QLabel(self.scroll_contents)
        labelImg1.move(20,160)
        labelImg1.resize(300,200)
        labelImg1.setStyleSheet(" border:4px;border-color:black;")
        labelImg1.setPixmap(QtGui.QPixmap('../testImage/labelImg1.png').scaled(labelImg1.width(),labelImg1.height()))

        label2 = QLabel(self.scroll_contents)
        label2.setText('二、基本原理')
        label2.move(10, 370)
        label2.resize(150, 20)
        label2.setStyleSheet(" border:4px;border-color:black;color:red")
        label2.setFont(QFont("Microsoft YaHei", 13))

        text2 = QTextEdit(self.scroll_contents)
        text2.setText("因为目标是把模糊图像IB在没有提供模糊核的情况下恢复成清晰图像IS，因此，我们需要训练一个CNN GθG作为生成器。每张IB都对应着一张估计出的清晰图像IS。此外，在训练阶段，我们将引入critic函数DθD，以对抗的方式训练两个网络。")
        text2.move(20, 400)
        text2.resize(700, 60)
        text2.setStyleSheet('background-color:white')
        text2.setReadOnly(True)

        labelImg2 = QLabel(self.scroll_contents)
        labelImg2.move(20, 470)
        labelImg2.resize(700, 200)
        labelImg2.setStyleSheet(" border:4px;border-color:black;")
        labelImg2.setPixmap(QtGui.QPixmap('../testImage/labelImg2.png').scaled(labelImg2.width(), labelImg2.height()))

        label3 = QLabel(self.scroll_contents)
        label3.setText('三、模型简图')
        label3.move(10, 680)
        label3.resize(150, 20)
        label3.setStyleSheet(" border:4px;border-color:black;color:red")
        label3.setFont(QFont("Microsoft YaHei", 13))

        labelImg3 = QLabel(self.scroll_contents)
        labelImg3.move(20, 720)
        labelImg3.resize(500, 500)
        labelImg3.setStyleSheet(" border:4px;border-color:black;")
        labelImg3.setPixmap(QtGui.QPixmap('../testImage/labelImg3.png').scaled(labelImg3.width(), labelImg3.height()))

        label4 = QLabel(self.scroll_contents)
        label4.setText('四、产品介绍')
        label4.move(10, 1240)
        label4.resize(150, 20)
        label4.setStyleSheet(" border:4px;border-color:black;color:red")
        label4.setFont(QFont("Microsoft YaHei", 13))

        text2 = QTextEdit(self.scroll_contents)
        text2.setText("对原有代码封装，可以实现输入模糊图片，选择权重模型，得到去模糊的图片，如下图所示效果：")
        text2.move(20, 1270)
        text2.resize(700, 40)
        text2.setStyleSheet('background-color:white')
        text2.setReadOnly(True)

        labelImg4 = QLabel(self.scroll_contents)
        labelImg4.move(20, 1320)
        labelImg4.resize(600, 300)
        labelImg4.setStyleSheet(" border:4px;border-color:black;")
        labelImg4.setPixmap(QtGui.QPixmap('../testImage/labelImg4.png').scaled(labelImg4.width(), labelImg4.height()))

        self.scroll_area.setWidget(self.scroll_contents)
        self.addWidget(self.scroll_area)


        op = QGraphicsOpacityEffect()
        op.setOpacity(1)

        self.setStyleSheet('background-color:rgb(0,0,0,1)')
        self.resize(800,800)

if __name__ == '__main__':
    app=QApplication(sys.argv)
    my=detail()
    my.show()
    sys.exit(app.exec_())
