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
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename


root=Tk()
root.geometry('240x400')

root.title('Visual Attendance')
lbl1 = Label(root, text="Visual Attendance",font=("Arial Bold", 19),padx=10)
lbl1.grid(column=0, row=0)
global Id, name

def reg_window():
    reg=Tk()
    reg.title('Register')
    reg.geometry('220x300')

    nameLabel=Label(reg,text="Enter Name", pady=5)
    idLabel = Label(reg,text="Enter ID", pady=5)

    nameLabel.grid(column=0,row=0)
    idLabel.grid(column=0,row=1)

    nameEntry = Entry(reg)
    nameEntry.grid(column=1,row=0)

    idEntry = Entry(reg)
    idEntry.grid(column=1,row=1)

    def get_details():
        Id = (idEntry.get())
        name = (nameEntry.get())
        return Id, name

    def TakeImages():

        Id,name = get_details()
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
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
            csvFile.close()

        print("images taken")
        success = Label(reg, text="Images taken", font=("Arial Bold", 15), padx=50)
        success.grid(column=0, row=7, columnspan=2)
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
        success = Label(reg, text="Images trained", font=("Arial Bold", 15), padx=50)
        success.grid(column=0, row=8, columnspan=2)


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

    empty = Label(reg, text="", pady=10)
    empty.grid(column=0, row=2)
    btnTakeImg = Button(reg, text='Take Image', command=TakeImages,height=1,width=20)
    btnTakeImg.grid(row=3, column=0, columnspan=3,padx=35, pady=5)



    btnTrainImg = Button(reg, text='Train Image', command=TrainImages,height=1,width=20,anchor=CENTER)
    btnTrainImg.grid(row=4, column=0, columnspan=2,padx=35, pady=5)

    btnClear = Button(reg, text='Clear', command=clear_reg,height=1,width=20,anchor=CENTER)
    btnClear.grid(row=5, column=0,columnspan=2,padx=35, pady=5)



    reg.mainloop()

def attend():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    at_details = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                at_details.loc[len(at_details)] = [Id, aa, date, timeStamp]


            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)

        at_details=at_details.drop_duplicates(subset=['Id'],keep='first')
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    at_details.to_csv("Attendance\Attendance.csv",index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    success = Label(root, text="Attendance Successful", font=("Arial Bold", 10), padx=50)
    success.grid(column=0, row=4, columnspan=2)
    print("success")

    #

def attend_details():

    details_window = Tk()
    details_window.title('Attendance details')
    width = 500
    height = 400
    screen_width = details_window.winfo_screenwidth()
    screen_height = details_window.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    details_window.geometry("%dx%d+%d+%d" % (width, height, x, y))
    details_window.resizable(0, 0)

    TableMargin = Frame(details_window, width=500)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, columns=("Id", "Name", "Date","Time"), height=400, selectmode="extended",
                        yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)
    tree.heading('Id', text="Id", anchor=W)
    tree.heading('Name', text="Name", anchor=W)
    tree.heading('Date', text="Date", anchor=W)
    tree.heading('Time', text="Time", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=100)
    tree.column('#2', stretch=NO, minwidth=0, width=100)
    tree.column('#3', stretch=NO, minwidth=0, width=100)
    tree.pack()

    with open('Attendance/Attendance.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:

            Id = row['Id']
            Name = row['Name']
            Date = row['Date']
            Time = row['Time']
            tree.insert("", 0, values=(Id,Name[2:-2], Date,Time))
        count=len(row["Id"])
        tree.insert("", 100, values=("Total attendance", count - 1))
    details_window.mainloop()


empty = Label(root, text="", font=("Arial Bold", 12), pady=50)
empty.grid(column=0, row=1, columnspan=2)
btn1=Button(root,text='Attend',height=1,width=20,command=attend,anchor=CENTER)
btn1.grid(row=2,column=0,padx=5, pady=5)
btn2=Button(root,text='Register',height=1,width=20,command=reg_window)
btn2.grid(row=3,column=0,padx=5, pady=5)

btn3=Button(root,text="Attendance Details",height=1,width=20,command =attend_details)
btn3.grid(row=5,column=0,padx=5, pady=5)

root.mainloop()
