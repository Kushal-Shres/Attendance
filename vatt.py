from tkinter import *
from tkinter import scrolledtext,messagebox
import tkinter as tk
from tkinter import Message, Text
import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time


root=Tk()
root.geometry('250x400')

root.title('Visual Attendance')
lbl1 = Label(root, text="Visual Attendance",font=("Arial Bold", 15),padx=50)
lbl1.grid(column=0, row=0)




def reg_window():
    reg=Tk()
    reg.title('Register')
    reg.geometry('250x400')

    nameLabel=Label(reg,text="Enter Name")
    idLabel = Label(reg,text="Enter ID")

    nameLabel.grid(column=0,row=0)
    idLabel.grid(column=0,row=1)

    nameEntry = Entry(reg)
    nameEntry.grid(column=1,row=0)

    idEntry = Entry(reg)
    idEntry.grid(column=1,row=1)


    def TakeImages():
        Id = (idEntry.get())
        name = (nameEntry.get())
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()

        print("images taken")
        # else:
        #     if (is_number(Id)):
        #         res = "Enter Alphabetical Name"
        #         message.configure(text=res)
        #     if (name.isalpha()):
        #         res = "Enter Numeric Id"
        #         message.configure(text=res)

    def clear_reg():
        idEntry.delete(0,END)
        nameEntry.delete(0,END)

    def TrainImages():
        recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("TrainingImage")
        recognizer.train(faces, np.array(Id))
        recognizer.save("TrainingImageLabel\Trainner.yml")
        res = "Image Trained"  # +",".join(str(f) for f in Id)
        message.configure(text=res)
        print("image trained")

    def getImagesAndLabels(path):
        # get the path of all the files in the folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # print(imagePaths)

        # create empth face list
        faces = []
        # create empty ID list
        Ids = []
        # now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(Id)
        return faces, Ids

    btnTakeImg = Button(reg, text='Take Image', padx=50, command=TakeImages)
    btnTakeImg.grid(row=3, column=0, columnspan=2)
    btnTrainImg = Button(reg, text='Train Image', padx=50, command=TrainImages)
    btnTrainImg.grid(row=4, column=0, columnspan=2)

    btnClear = Button(reg, text='Clear', padx=50, command=clear_reg)
    btnClear.grid(row=5, column=0,columnspan=2)

    reg.mainloop()



btn1=Button(root,text='Attend',padx=50,command="")
btn1.grid(row=1,column=0)
btn2=Button(root,text='Register',padx=50,command=reg_window)
btn2.grid(row=2,column=0)


root.mainloop()
