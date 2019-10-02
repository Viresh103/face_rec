#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:07:02 2019

@author: tesla
"""

import cv2
import os
import numpy as np

def datacreater(name):
    
    a=[]
    for filename in os.listdir(name):
        img=cv2.imread(name+"/"+filename)
        print(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img=cv2.resize(gray, (64,64), interpolation = cv2.INTER_AREA)
        
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        net=norm_image.flatten()
        a.append(net)
    return a  
data1=datacreater("./data1")
y1=np.array([1 for i in range(116)])
#data1 it is the directory where class 0 images are stored
data=datacreater("./data")
y=np.array([0 for i in range(738)])
#data it is the directory where class 1 images are stored
df=np.vstack((data1,data))
#############################################

from sklearn.decomposition import PCA
pca=PCA(n_components=7)
pca.fit(df)
###################################
data1=pca.transform(data1)
data1=np.hstack((data1,np.resize(y1,(116,1))))
data=pca.transform(data)
data=np.hstack((data,np.resize(y,(738,1))))
df=np.vstack((data1,data))
####################################################### 
x=df[:,0:7]
y=df[:,7:8]
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x,y)

######################################################################  
