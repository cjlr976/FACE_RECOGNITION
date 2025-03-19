import cv2
import numpy as np
import face_recognition

#Load images as rgb
imgJane = face_recognition.load_image_file('ImageBasic/rachel_blais.jpg')
imgJane = cv2.cvtColor(imgJane,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/rachel_blais_test.jpg')
imgTest = cv2. cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgJane)[0]
encodeJane = face_recognition.face_encodings(imgJane)[0]
cv2.rectangle(imgJane,(faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeJane], encodeTest)
faceDis = face_recognition.face_distance([encodeJane], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results}, {round(faceDis[0],2)}',(50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Jane', imgJane)
cv2.imshow('Jane Test', imgTest)
cv2.waitKey(0)