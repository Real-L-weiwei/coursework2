from PyQt5 import uic, QtWidgets,QtGui
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import * 

#Import the ui into the class
class LoggedInWindow(QMainWindow,uic.loadUiType("loggedInMenu.ui")[0]):
    def __init__(self,appWindow):
        #Inheritance
        super().__init__()
        #sets up the ui
        self.setupUi(appWindow)
        self.AppWindow = appWindow
        print(self.AppWindow.userInfo)
        text = "Hello " + str(self.AppWindow.userInfo[1])
        self.user_label.setText(text)

        self.start_but.clicked.connect(self.loadQuestionWindow)

    def loadQuestionWindow(self):
        self.AppWindow.setupQuestionWindow(self.AppWindow.userInfo)
        
        

