# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:15:14 2020

@author: Harshini Priya
"""

import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('face.jpg')
fix_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(fix_img,1.3,5)

for (x,y,w,h) in faces:
    img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    
cv2.imshow("Face detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

