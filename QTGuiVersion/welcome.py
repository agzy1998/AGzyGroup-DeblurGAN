import sys
import time
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QMainWindow,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QGraphicsOpacityEffect,QSplashScreen,QScrollArea
from PyQt5.QtGui import QFont,QPalette,QPainter,QIcon,QCursor,QMouseEvent
from PyQt5 import QtGui,QtCore
from PyQt5.QtCore import Qt,QSize
from QTGuiVersion.deglurGanDIY import myWidget
from QTGuiVersion.deblurGanDetail import detail
from QTGuiVersion.deblurTrain import DeblurTrain
from QTGuiVersion.deblurGanAbout import about

app = QApplication(sys.argv)
my1=detail()
my2=myWidget()
#my3=DeblurTrain()
my4=about()
class Welcome(QWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.stack=QStackedWidget()
        #self.stack.addWidget(QLabel('ksdjfkdjf'))
        self.stack.addWidget(my1)
        self.stack.addWidget(my2)
        #self.stack.addWidget(my3)
        self.stack.addWidget(my4)



        hbox = QHBoxLayout()
        self.left = QListWidget()
        # QFrame 控件添加StyledPanel样式能使QFrame 控件之间的界限更加明显
        #left.setFrameShape(QFrame.StyledPanel)
        self.right = self.stack

        self.left.setMinimumSize(300,800)
        self.left.setMaximumSize(300,800)


        self.left.setStyleSheet('QListWidget{border:4px solid; border-color:white;color:blue;border-radius:20px;background-color:rgb(0,0,0,1)}' 'QListWidget::Item{padding-top:20px; padding-bottom:4px;border:4px solid;border-radius:15px}''QListWidget::Item:hover{background:skyblue; }''QListWidget::item:selected{background:lightgray; color:red;}' 'QListWidget::item:selected:!active{border-width:0px; background:lightgreen;}')
        self.left.setFont(QFont("Roman times",18,QFont.Bold))

        #self.right.setStyleSheet('QStackedWidget{border:4px solid; color:blue;border-radius:20px;}')



        #right.setFrameShape(QFrame.StyledPanel)
        hbox.addWidget(self.left)
        hbox.addWidget(self.right)
        self.setLayout(hbox)


        self.left.insertItem(0,'deblurGan简介')
        self.left.insertItem(1,'deblurGanDIY')
        #self.left.insertItem(2,'DeblurTrain')
        self.left.insertItem(2,'关于我们')


        #self.stack.setCurrentIndex(1)
        self.left.currentRowChanged.connect(self.switchCover)



        palette=QPalette()
        palette.setBrush(self.backgroundRole(),QtGui.QBrush(QtGui.QPixmap('../testImage/index.jpeg').scaled(1100,800)))
        self.setPalette(palette)



        self.closeBtn=QPushButton(self)
        self.closeBtn.move(1050,20)
        self.closeBtn.clicked.connect(self.close)

        pixmap=QtGui.QPixmap("../testImage/button2.png").scaled(20,20)


        self.closeBtn.setIcon(QIcon("../testImage/button2.png"));
        self.closeBtn.setIconSize(QSize(20,20));
        self.closeBtn.setMask(pixmap.mask());
        self.closeBtn.setFixedSize(QSize(20,20));


        self.shrinkBtn=QPushButton(self)
        self.shrinkBtn.move(1020,16)
        self.shrinkBtn.clicked.connect(self.showMinimized)

        pixmap2=QtGui.QPixmap('../testImage/shrinkBtn.png').scaled(30,30)

        self.shrinkBtn.setIcon(QIcon('../testImage/shrinkBtn.png'))
        self.shrinkBtn.setIconSize(QSize(30,30))
        self.shrinkBtn.setMask(pixmap2.mask())
        self.shrinkBtn.setFixedSize(QSize(30,30))

        #self.setWindowOpacity(0.9)
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)


        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.move(500, 100)
        self.resize(1100,800)
        self.show()


    def mousePressEvent(self, a0: QtGui.QMouseEvent) :
        if a0.button()==Qt.LeftButton:
            self.m_flag=True
            self.m_Position=a0.globalPos()-self.pos()
            a0.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
            
    def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(a0.globalPos()-self.m_Position)
            a0.accept()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) :
        self.m_flag=False
        self.setCursor(QCursor(Qt.ArrowCursor))


    def switchCover(self,i):
        self.stack.setCurrentIndex(i)
        self.update()
    





if __name__ == '__main__':
    #app = QApplication(sys.argv)
    splash=QSplashScreen(QtGui.QPixmap('../testImage/splash.png'))

    splash.show()

    #QApplication.processEvents()
    #time.sleep(3)


    step=1
    while step<=4:
        QApplication.processEvents()
        time.sleep(1)
        step+=1




    wel=Welcome()
    splash.finish(wel)

    sys.exit(app.exec_())