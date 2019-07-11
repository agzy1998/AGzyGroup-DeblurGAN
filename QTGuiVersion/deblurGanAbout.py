import sys
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QGraphicsOpacityEffect
from PyQt5.QtGui import QFont,QPalette
from PyQt5.QtCore import Qt



class about(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def  init(self):
        self.label=QLabel(self)
        self.label.resize(800,800)
        self.label.setText('组长：高振宇\n组员：刘馨 柴乐铭 田森\n联系我们：agzy1998@163.com')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Roman times", 18, QFont.Bold))


        pe = QPalette()
        pe.setColor(QPalette.WindowText, Qt.red)  # 设置字体颜色
        self.label.setPalette(pe)


        #self.setStyleSheet('border:4px solid;border-color:white;border-radius:20px;')

        self.resize(800,800)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my = about()
    my.show()
    sys.exit(app.exec_())
