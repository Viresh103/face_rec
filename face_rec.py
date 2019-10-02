#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:58:07 2019

@author: tesla
"""

#from sklearn.decomposition import PCA
#pca=PCA(n_components=8)
#pca.fit(data)
#import pickle 
import numpy as np
import  cv2


FACE_CLASSIFIER = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

SCALE_FACTOR = 1.3
BLUE_COLOR = (255, 0, 0)
MIN_NEIGHBORS = 5

cap = cv2.VideoCapture(0)
count=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    faces = FACE_CLASSIFIER.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)
      
    font=cv2.FONT_HERSHEY_DUPLEX
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        img=cv2.resize(gray, (64,64), interpolation = cv2.INTER_AREA)
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        net=norm_image.flatten() 
        net=pca.transform([net])
        #real1=pca.transform([net])
#        print((np.dot((real-real1),(real-real1).T))**0.5 )
        #if (np.dot((real-real1),(real-real1).T))**0.5 <15:
        #    cv2.putText(frame,"viresh",(x+6,y-6),font,0.5,(255,255,255),1)
        if classifier.predict(net)[0]==1:
            cv2.putText(frame,"viresh",(x+6,y-6),font,0.5,(255,255,255),1)
       
    cv2.imshow('img',frame)
    if cv2.waitKey(10)& 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()