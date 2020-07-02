# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:20:19 2020

@author: Harshini Priya
"""

import cv2

pedestrian_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('Pedestrian.mp4')
while True:
    ret,frame = cap.read()
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies = pedestrian_classifier.detectMultiScale(img,1.2,2)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    cv2.imshow("Pedestrian detection",frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()