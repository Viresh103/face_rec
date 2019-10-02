#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:26:03 2019

@author: tesla
"""

import cv2
import numpy as np

FACE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SCALE_FACTOR = 1.3
BLUE_COLOR = (255, 0, 0)
MIN_NEIGHBORS = 5

cap = cv2.VideoCapture(0)
count=1.13
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    faces = FACE_CLASSIFIER.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)
      
   
    for (x,y,w,h) in faces:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = frame[y:y+h, x:x+w]
      
       cv2.imwrite("./data/image"+str(count)+".jpg",roi_color)
       count+=1
       
    cv2.imshow('img',frame)
    if cv2.waitKey(10)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
                   