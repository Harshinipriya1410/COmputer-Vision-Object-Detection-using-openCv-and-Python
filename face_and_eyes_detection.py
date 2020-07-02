# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:54:19 2020

@author: Harshini Priya
"""

import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('faceandeye.jpg')
fix_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


faces = face_classifier.detectMultiScale(fix_img,1.3,5)
eyes = eye_classifier.detectMultiScale(fix_img,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(125,255,0),3)
for (x,y,w,h) in eyes:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,125,255),3)
    
cv2.imshow("Face and eye detection",image)
cv2.waitKey(0)
cv2.destroyAllWindows()