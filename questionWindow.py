import random
import sqlite3

from PyQt5 import uic, QtWidgets,QtGui
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
#Load image
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import * 

#Play audio
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from dictfactory import *
from completeRecordProcessSound import *
from train4 import *



class QuestionWindow(QMainWindow,uic.loadUiType("question.ui")[0]):
    def __init__(self,appWindow):
        #Initialising variables
        ####
        self.answered = False
        self.maxQs = 5
        self.answeredQs = 0
        ####
        self.qnum=1
        self.accuracy = 0.0
        super().__init__()
        #sets up the ui
        self.setupUi(appWindow)
        #Extract user info from parameters passed in
        self.AppWindow = appWindow
        print(self.AppWindow.userInfo)
        #If guest uses program
        if self.AppWindow.userInfo == False:
            self.guest = True
        #Extract info into local class variables if user isn't a guest
        else:
            self.guest=False
            self.userid = self.AppWindow.userInfo[0]
            self.username = self.AppWindow.userInfo[1]
        
        ####
        #Hide the play button as audio can't be played before attempt
        self.play_but.hide()
        #Set up an instance of the media player
        self.player = QMediaPlayer()
        ####
        
        
        self.conn = sqlite3.connect("playerQuestions.db",isolation_level=None)
        self.conn.execute("PRAGMA foreign_keys = 1")
        self.conn.row_factory = dict_factory
        
        #blancks out the label
        self.img_label.setText("")
        #Sets recording status to off at start (displayed)
        #Changes label to output correct question number
        tempQ = "Question:" + str(self.qnum)
        self.qnum_label.setText(tempQ)
        #scales the picture
        self.img_label.setScaledContents(True)
        

        #Fetch all the name of pictures from database
        self.pictures = self.fetchPic()
        
        #Extract relevant information from the dictionary, 
        #after the function returns the dictionary with the right word
        self.currentPic = self.choosePic()
        self.picName = self.currentPic.get("word")
        self.picJpg = self.currentPic.get("jpg")
        
        #pixMap maps the image into the label
        self.img_label.setPixmap(QtGui.QPixmap(self.picJpg))
        
        #SQL for non-guests
        if self.guest != True:
            c = self.conn.cursor()
            c.execute("BEGIN")
            c.execute('DELETE FROM userQuestion')
            insert = 'INSERT INTO userQuestion(questid,word,userid,accuracy,answered) VALUES(:qn,:p,:n,null,null)'
            c.execute(insert,{'qn':self.qnum,'p':self.picName,'n':self.userid})
            c.execute("COMMIT")
        #SQL for guests
        else:
            c = self.conn.cursor()
            c.execute("BEGIN")
            c.execute('DELETE FROM userQuestion')
            insert = 'INSERT INTO userQuestion(questid,word,userid,accuracy,answered) VALUES(:qn,:p,null,null,null)'
            c.execute(insert,{'qn':self.qnum,'p':self.picName})
            c.execute("COMMIT")
        
        #Record button clicked
        self.rec_but.clicked.connect(self.record)
        
        #Confirm recording button clicked
        self.confirm_but.clicked.connect(self.confirm)

        #Action for clicking the next button
        self.nextQ_but.clicked.connect(self.nextQ)
        
        ####
        #Action for clicking the "I don't know button"
        self.idk_box.clicked.connect(self.idk)
        
        #Action for play button clicked
        self.play_but.clicked.connect(self.play)
        ####
        

    def fetchPic(self):
        c = self.conn.cursor()
        select = 'SELECT pictureid,word,jpg FROM pictures'
        c.execute(select)
        pictures = c.fetchall()
        print(pictures)
        return pictures
    
    #Method for randomly choosing a picture
    def choosePic(self):
        length = len(self.pictures)-1
        num = random.randint(0,length)
        print(self.pictures[num])
        return self.pictures[num]

    def record(self):
        #Clear feedback label
        self.feed_label.setText("Feedback")
        
        self.userIssue = soundMain(self.qnum)
        #Check for any issues during recording
        if self.userIssue == True:
            #Change the content of the label to notify user
            self.rec_label.setText("Recording Status: Done \n [Input too quiet, try again]")
        else:
            #Notify the user that the recording has finished without any problems
            self.rec_label.setText("Recording Status: Done \n [No issues]")
        
    def confirm(self):
        #try and except statement to ctach out an errors
        try:
            #info = [real_accuracy,highest likely word,fake_accuracy]
            info = mainConfirm(self.picName,self.qnum)
        #Only possible error if the user forgot to record
        except:
            self.feed_label.setText("You can't confirm \n your answer without \n recording (inputting!)")
            #Quit the function to alt any advancements
            return None
        
        #output accuracy of what the word should have said is
        self.accuracy = str(round(info[0]*100,2))
        tempAcc = "Accuracy: "+self.accuracy+"%"
        self.acc_label.setText(tempAcc)
        #Output the actual word represented by the image
        correct = "Correct word: " + str(self.picName)
        self.correct_label.setText(correct)
        #Feedback to the user
        #\n inserted to create line breaks
        if info[0] != info[2]:
            tempFeed1 = "The algorithm \n the most likely \n word spoken was\n'"+str(info[1])+"' at "+str(round(info[2]*100,2))+"% accuracy"
            self.feed_label.setText(tempFeed1)
        else:
            tempFeed2 = "Good accuracy! \n The algorithm \n thinks that was \n the word most \n likely spoken"
            self.feed_label.setText(tempFeed2)
            
        #SQL for inserting the value of accuracy
        c = self.conn.cursor()
        c.execute("BEGIN")
        update = 'UPDATE userQuestion SET accuracy = :acc WHERE questid = :qn'
        c.execute(update,{'acc':self.accuracy,'qn':self.qnum})
        
        #SQL for setting the answered column to true for this question
        update = 'UPDATE userQuestion SET answered = :ans WHERE questid = :qn'
        c.execute(update,{'ans':"true",'qn':self.qnum})
        c.execute("COMMIT") 
    
        ####
        #Hide the "Confirm Recording" button so it can't be clicked again
        self.confirm_but.hide()
        #Show the play button for the user to check
        self.play_but.show()
        #Hide the record button so it can't be clicked again
        self.rec_but.hide()
        #Hide the "I don't know" box so it can't be clicked
        self.idk_box.hide()
        #Indicate the answer has been checked
        self.answered = True
        ####

####
    def idk(self):
        #Output the actual word represented by the image
        correct = "Correct word: " + str(self.picName)
        self.correct_label.setText(correct)

        #Hide the "Confirm Recording" button so it can't be clicked
        self.confirm_but.hide()
        #Show the play button for the user to check
        self.play_but.show()
        #Hide the record button so it can't be clicked
        self.rec_but.hide()
        #Hide the "I don't know" box so it can't be clicked again
        self.idk_box.hide()
        
        #SQL for setting the answered column to false for this question
        c = self.conn.cursor()
        c.execute("BEGIN")
        update = 'UPDATE userQuestion SET answered = :ans WHERE questid = :qn'
        c.execute(update,{'ans':"false",'qn':self.qnum})
        c.execute("COMMIT") 
####       
        
    def play(self):
        #Get the file path for the correct audio file
        init_filepath = 'correct/' + str(self.picName) + '.wav'
        filepath = os.path.join(os.getcwd(),init_filepath)
        #Convert it into URL
        url = QUrl.fromLocalFile(filepath)
        #Load the content of the file
        content = QMediaContent(url)
        
        #Lood the content into the instance and play the audio
        self.player.setMedia(content)
        self.player.play()
        
        
        
    #label replaced by a random picture
    def nextQ(self):
        ####
        #Check if the number of questions for the session has been reached
        if self.qnum >= self.maxQs:
            #Check if the answer has been submitted
            if self.answered == True:
                #Increment counter for the number of questions answered
                self.answeredQs = self.answeredQs + 1
            #Navigate to summary page
            self.AppWindow.setupSummaryWindow(self.AppWindow.userInfo,self.answeredQs,self.qnum)
        else:
            
            #Check if the answer has been submitted
            if self.answered == True:
                #Increment counter for the number of questions answered
                self.answeredQs = self.answeredQs + 1
                    
            #Check if the answer has been submitted or the "I don't know" checkbox ticked
            if self.idk_box.isChecked() == False and self.answered == False:
                #Display a message to remind the user 
                self.launchPopup("You have not answered the question")
            else:
                #Show the "Confirm Recording" button for the next question
                self.confirm_but.show()
                #Show the "Recording" button for the next question
                self.rec_but.show()
                #Show the "I don't know" box for the next question
                self.idk_box.show()
                #Hide the play button as audio can't be played before attempt
                self.play_but.hide()
                #Reset the "I don't know" checkbox
                self.idk_box.setChecked(False)
                #Indicate new question hasn't been answered
                self.answered= False
            ####
                #Changing question number and outputting
                self.qnum = self.qnum + 1
                tempQ = "Question:" + str(self.qnum)
                self.qnum_label.setText(tempQ)

                #Extract relevant information from the dictionary, 
                #after the function returns the dictionary with the right word
                self.currentPic = self.choosePic()
                self.picName = self.currentPic.get("word")
                self.picJpg = self.currentPic.get("jpg")

                #pixMap maps the image into the label
                self.img_label.setPixmap(QtGui.QPixmap(self.picJpg))
            
                #Setting all labels to blank
                self.acc_label.setText("Accuracy:")
                self.correct_label.setText("Correct word:")
                self.feed_label.setText("Feedback")
                self.rec_label.setText("Recording Status: Off")

                #SQL for non-guests
                if self.guest != True:
                    c = self.conn.cursor()
                    c.execute("BEGIN")
                    insert = 'INSERT INTO userQuestion(questid,word,userid,accuracy,answered) VALUES(:qn,:p,:n,null,null)'
                    c.execute(insert,{'qn':self.qnum,'p':self.picName,'n':self.userid})
                    c.execute("COMMIT")
                #SQL for guests
                else:
                    c = self.conn.cursor()
                    c.execute("BEGIN")
                    insert = 'INSERT INTO userQuestion(questid,word,userid,accuracy,answered) VALUES(:qn,:p,null,null,null)'
                    c.execute(insert,{'qn':self.qnum,'p':self.picName})
                    c.execute("COMMIT")
        
    ####
    #Pass the text in to launch and setup the popup box
    def launchPopup(self,text):
        pop = Popup(text,self)
        pop.show()
    ####
        


    
#### 
#QDialog used as the type of PyQt widget, load the file
class Popup(QDialog,uic.loadUiType("popup.ui")[0]):
    def __init__(self,name,parent):
        #Inheritance
        super().__init__(parent)
        #Set the styling
        self.setStyleSheet('font-size: 20px; font-family: calibri')
        #Set up dimensions for the box
        self.resize(400,100)
        #Replace content in the label with intended content
        self.label = QLabel(name,self)
####


