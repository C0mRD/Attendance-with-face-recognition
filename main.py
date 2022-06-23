import numpy as np
import cv2
import face_recognition
import os
import datetime as dt

Path = 'Images'
images = []
classNames = []
imgList = os.listdir(Path)
for img in imgList:
    curImg = cv2.imread(f'{Path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0])

def FindEncoding(images):
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

def MarkAttendence(name):
    with open('Attendence.csv','r+') as f:
        data = f.readlines()
        nameList = []
        date = dt.date.today()
        for line in data:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            cTime = dt.datetime.now()
            dtString = cTime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date},{dtString}')


encodeKnownList = FindEncoding(images)
cap = cv2.VideoCapture(0)
while True:
    _,image = cap.read()
    imageS = cv2.resize(image,(0,0),None,0.25,0.25)
    imageS = cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(imageS)
    curEncodings = face_recognition.face_encodings(imageS,faces)
    for encodeFace,FaceLoc in zip(curEncodings,faces):
        matches = face_recognition.compare_faces(encodeKnownList,encodeFace)
        faceDist = face_recognition.face_distance(encodeKnownList,encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1= FaceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MarkAttendence(name)

    cv2.imshow("Attendence System",image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
