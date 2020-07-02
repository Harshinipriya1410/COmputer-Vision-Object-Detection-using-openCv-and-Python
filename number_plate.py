# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:43:12 2020

@author: Harshini Priya
"""

import cv2

plate_classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

image = cv2.imread('bmw.png')
fix_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

plates = plate_classifier.detectMultiScale(fix_img,1.3,4)

for (x,y,w,h) in plates:
    img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
    
cv2.imshow("Number plate detection",img)
cv2.waitKey(0)
cv2.destroyAllWindows()