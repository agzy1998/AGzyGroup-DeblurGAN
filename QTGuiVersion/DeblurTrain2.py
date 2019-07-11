import sys

from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QFileDialog,QLabel,QMainWindow,QHBoxLayout,QFrame,QSplitter,QListWidget,QListWidgetItem,QStackedWidget,QGraphicsOpacityEffect,QMessageBox
from PyQt5.QtGui import QFont,QPalette
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from scripts.deblur_image2  import myDeblur


class myTrain(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.init()

