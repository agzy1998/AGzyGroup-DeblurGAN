import sys

from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QMainWindow,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QGraphicsOpacityEffect,QMessageBox
from PyQt5.QtGui import QFont,QPalette
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from scripts.deblur_image2  import myDeblur


class myWidget(QStackedWidget):
    selectBtn=0
    deblurBtn=0
    originalLabel=0
    deblurLabel=0

    imgNameIn=0#获取图片的路径
    imgNameOut=0

    modelName=0

    mydeblur=myDeblur()
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.selectBtn=QPushButton('Select image',self)
        self.selectBtn.move(50,50)
        self.selectBtn.resize(300,30)
        self.selectBtn.clicked.connect(self.openImage)
        self.selectBtn.setStyleSheet('QPushButton {  border-width:2px;border-color:rgb(10,45,110);border-radius:10px;color:red; border-style:outset; font:bold 15px ;  }')

        self.deblurBtn=QPushButton('Deblur',self)
        self.deblurBtn.move(50,410)
        self.deblurBtn.resize(300,30)
        self.deblurBtn.clicked.connect(self.deblurImage)
        self.deblurBtn.setStyleSheet('QPushButton {  border-width:2px;border-color:rgb(10,45,110);border-radius:10px;color:red; border-style:outset; font:bold 15px;  }')

        self.originalLabel=QLabel(self)
        self.originalLabel.move(50,90)
        self.originalLabel.resize(300,300)
        self.originalLabel.setStyleSheet("border:2px solid;background-color:white;")

        self.deblurLabel=QLabel(self)
        self.deblurLabel.move(50,470)
        self.deblurLabel.resize(600,300)
        self.deblurLabel.setStyleSheet("border:2px solid;background-color:white;")

        self.modelBtn=QPushButton('Select Model',self)
        self.modelBtn.move(400,50)
        self.modelBtn.resize(100,100)
        self.modelBtn.setStyleSheet('QPushButton {  border-width:2px;border-color:rgb(10,45,110);border-radius:10px;color:red; border-style:outset; font:bold 15px;  }')
        self.modelBtn.clicked.connect(self.selectModel)
        '''
        palette = QPalette()
        palette.setBrush(self.backgroundRole(),QtGui.QBrush(QtGui.QPixmap('/home/liuxin/下载/lion.jpg').scaled(800, 800)))
        self.setPalette(palette)

        '''

        op = QGraphicsOpacityEffect()
        op.setOpacity(1)



        #self.setStyleSheet('border:4px solid;border-color:white;border-radius:20px;')
        self.resize(800,800)
        #self.setGeometry(100,100,800,800)

    def selectModel(self):
        self.modelName,modelType=QFileDialog.getOpenFileName(self, "Open Image", "/home/liuxin/PycharmProjects/tets/weights", "*.h5")

    def openImage(self):
        self.imgNameIn, imgType = QFileDialog.getOpenFileName(self, "Open Image", "/home/liuxin/PycharmProjects/tets/testImage", "*.jpeg;;*.jpg;;*.png;;All Files(*)")
        print(self.imgNameIn,imgType)
        jpg = QtGui.QPixmap(self.imgNameIn).scaled(self.originalLabel.width(), self.originalLabel.height())
        self.originalLabel.setPixmap(jpg)

    def deblurImage(self):
        if self.imgNameIn==0:
            QMessageBox.about(self,'warning','Have not select image')
            return
        if self.modelName==0:
            QMessageBox.about(self, 'warning', 'Have not select model')
            return
        if self.imgNameIn!=0:
            #self.mydeblur.deblur(self.imgNameIn)
            self.imgNameOut=self.mydeblur.deblur(self.imgNameIn,self.modelName)
            print(self.imgNameOut)
            jpg = QtGui.QPixmap(self.imgNameOut).scaled(self.deblurLabel.width(), self.deblurLabel.height())
            self.deblurLabel.setPixmap(jpg)









if __name__ == '__main__':
    app=QApplication(sys.argv)
    my=myWidget()
    my.show()
    sys.exit(app.exec_())

