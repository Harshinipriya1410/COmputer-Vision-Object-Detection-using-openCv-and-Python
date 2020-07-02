# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:02:24 2020

@author: Harshini Priya
"""

import cv2

car_classifier = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture('carvideo.mp4')
while True:
    ret,frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(img,1.2,3)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Car detection",frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()