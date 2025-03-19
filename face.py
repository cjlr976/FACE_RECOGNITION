import cv2
import numpy as np
import face_recognition

#Load images as rgb
imgJane = face_recognition.load_image_file('ImageBasic/JaneTrain.jpg')
imgJane = cv2.cvtColor(imgJane.cv2.COLOR_BGR2RGB)

cv2.imshow('Jane', imgJane)
cv2.waitKey(0)

conda rename -p C:\Users\chloe\GitHubProjects\FACE_RECOGNITION\.conda face_recog_env

conda rename --name .conda face_recog_env
