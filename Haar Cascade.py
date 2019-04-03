#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import itertools


# In[3]:


def haarCalculator(integralImage,origin,width,height,haarType,reverse):
    origin['first'] += 1
    origin['second'] += 1
    if width > integralImage.shape[0] or height > integralImage.shape[1] or origin['first'] < 1 or origin['second'] <1:
        return 0
    uLeft1= {}
    uRight1= {}
    lLeft1= {}
    lRight1= {}
    uLeft2= {}
    uRight2= {}
    lLeft2= {}
    lRight2= {}
    uLeft3= {}
    uRight3= {}
    lLeft3 = {}
    lRight3 = {}
    if(haarType == 0):
        uLeft1['first'] = int(origin['first'] - 1)
        uLeft1['second'] = int(origin['second'] - 1)

        uRight1['first'] = int(origin['first'] - 1)
        uRight1['second'] = int(origin['second'] + width - 1)
        lLeft1['first'] = int(origin['first'] + height - 1)
        lLeft1['second'] = int(origin['second'] - 1)

        lRight1['first'] = int(origin['first'] + height - 1)
        lRight1['second'] = int(origin['second'] + width - 1)

        uLeft2['first'] = int(origin['first'] - 1)
        uLeft2['second'] = int(origin['second'] + width - 1)

        uRight2['first'] = int(origin['first'] - 1)
        uRight2['second'] = int(origin['second'] + 2 * width - 1)

        lLeft2['first'] = int(origin['first'] + height - 1)
        lLeft2['second'] = int(origin['second'] + width - 1)

        lRight2['first'] = int(origin['first'] + height - 1)
        lRight2['second'] = int(origin['second'] + 2 * width - 1)
        firstFilter = integralImage[lRight1['first'], lRight1['second']] + integralImage[uLeft1['first'], uLeft1['second']] - (integralImage[lLeft1['first'], lLeft1['second']] + integralImage[uRight1['first'], uRight1['second']])
        secondFilter = integralImage[lRight2['first'], lRight2['second']] + integralImage[uLeft2['first'], uLeft2['second']] - (integralImage[lLeft2['first'], lLeft2['second']] + integralImage[uRight2['first'], uRight2['second']])
        if  not(reverse):
            convValue = firstFilter - secondFilter
        else:
            convValue = secondFilter - firstFilter
    elif haarType == 1:
        uLeft1['first'] = int(origin['first'] - 1)
        uLeft1['second'] = int(origin['second'] - 1)

        uRight1['first'] = int(origin['first'] - 1)
        uRight1['second'] = int(origin['second'] + width - 1)

        lLeft1['first'] = int(origin['first'] + height - 1)
        lLeft1['second'] = int(origin['second'] - 1)

        lRight1['first'] = int(origin['first'] + height - 1)
        lRight1['second'] = int(origin['second'] + width - 1)

        uLeft2['first'] = int(origin['first'] + height - 1)
        uLeft2['second'] = int(origin['second'] - 1)

        uRight2['first'] = int(origin['first'] + height - 1)
        uRight2['second'] = int(origin['second'] + width - 1)

        lLeft2['first'] = int(origin['first'] + 2 * height - 1)
        lLeft2['second'] = int(origin['second'] - 1)

        lRight2['first'] = int(origin['first'] + 2 * height - 1)
        lRight2['second'] = int(origin['second'] + width - 1)

        firstFilter = integralImage[lRight1['first'], lRight1['second']] + integralImage[uLeft1['first'], uLeft1['second']] - (integralImage[lLeft1['first'], lLeft1['second']] + integralImage[uRight1['first'], uRight1['second']])
        secondFilter = integralImage[lRight2['first'], lRight2['second']] + integralImage[uLeft2['first'], uLeft2['second']] - (integralImage[lLeft2['first'], lLeft2['second']] + integralImage[uRight2['first'], uRight2['second']])
        if not(reverse):
            convValue = firstFilter - secondFilter
        else:
            convValue = secondFilter - firstFilter
    if haarType == 2:
        uLeft1['first'] = int(origin['first'] - 1)
        uLeft1['second'] = int(origin['second'] - 1)

        uRight1['first'] = int(origin['first'] - 1)
        uRight1['second'] = int(origin['second'] + width - 1)

        lLeft1['first'] = int(origin['first'] + height - 1)
        lLeft1['second'] = int(origin['second'] - 1)

        lRight1['first'] = int(origin['first'] + height - 1)
        lRight1['second'] = int(origin['second'] + width - 1)

        uLeft2['first'] = int(origin['first'] - 1)
        uLeft2['second'] = int(origin['second'] + width - 1)

        uRight2['first'] = int(origin['first'] - 1)
        uRight2['second'] = int(origin['second'] + 2 * width - 1)

        lLeft2['first'] = int(origin['first'] + height - 1)
        lLeft2['second'] = int(origin['second'] + width - 1)

        lRight2['first'] = int(origin['first'] + height - 1)
        lRight2['second'] = int(origin['second'] + 2 * width - 1)


        uLeft3['first'] = int(origin['first'] - 1)
        uLeft3['second'] = int(origin['second'] + 2 * width - 1)

        uRight3['first'] = int(origin['first'] - 1)
        uRight3['second'] = int(origin['second'] + 3 * width - 1)

        lLeft3['first'] = int(origin['first'] + height - 1)
        lLeft3['second'] = int(origin['second'] + 2 * width - 1)

        lRight3['first'] = int(origin['first'] + height - 1)
        lRight3['second'] = int(origin['second'] + 3 * width - 1)

        firstFilter = integralImage[lRight1['first'], lRight1['second']] + integralImage[uLeft1['first'], uLeft1['second']] - (integralImage[lLeft1['first'], lLeft1['second']] + integralImage[uRight1['first'], uRight1['second']])
        secondFilter = integralImage[lRight2['first'], lRight2['second']] + integralImage[uLeft2['first'], uLeft2['second']] - (integralImage[lLeft2['first'], lLeft2['second']] + integralImage[uRight2['first'], uRight2['second']])
        thirdFilter = integralImage[lRight3['first'], lRight3['second']] + integralImage[uLeft3['first'], uLeft3['second']] - (integralImage[lLeft3['first'], lLeft3['second']] + integralImage[uRight3['first'], uRight3['second']])

        if not(reverse):
            convValue = firstFilter - secondFilter + thirdFilter
        else:
            convValue = secondFilter - firstFilter + thirdFilter
    origin['first'] -= 1
    origin['second'] -= 1
    return convValue

class StrongClassifier(object):
    def __init__(self,weakClassifiers,threshold):
        self.weakClassifiers = weakClassifiers
        self.threshold = threshold
    def predict(self,integralImages):
        indices = []
        for i in range(len(integralImages)):
            predict = 0
            for j in range(len(self.weakClassifiers)):
                score = haarCalculator(integralImages[i],self.weakClassifiers[j].origin
                        ,self.weakClassifiers[j].width,self.weakClassifiers[j].height,
                        self.weakClassifiers[j].haarType,0)
                predict += self.weakClassifiers[j].weight * (score*self.weakClassifiers[j].polarity > self.weakClassifiers[j].threshold * self.weakClassifiers[j].polarity)
            if(predict > self.threshold):
                indices.append(i)
        print(indices)
        return indices
class Feature(object):
    def __init__(self,x,y,width,height,haarType,threshold,polarity,weight):
        self.origin = {'first':x,'second':y}
        self.width = width
        self.height = height
        self.haarType = haarType
        self.threshold = threshold
        self.polarity = polarity
        self.weight = weight


# In[4]:


def integralImage(images):
    integralImages = []
    for i in images:
        intergral_image = cv2.integral(i)
        integralImages.append(intergral_image)
    return integralImages


# In[5]:


with open("strongClassifier.txt",'r') as txt:
    num = txt.readline()
    features = []
    for i in range(int(num)):
        sif = txt.readline().split(" ")
        feature = Feature(float(sif[0]),float(sif[1])
            ,float(sif[2]),float(sif[3]),float(sif[4]),float(sif[5]),float(sif[6]),float(sif[7]))
        features.append(feature)                                                                
    cass = StrongClassifier(features,float(txt.readline()))
print(cass.threshold)


# In[26]:


cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(5,5))
    r=255
    g=255
    b = 255
    cropped= gray[80:400,160:480]
    FaceOrNoFace = cass.predict([cv2.integral(cv2.resize(cropped,(24,24)))])
    if(len(FaceOrNoFace)==0):
        g=0
    cv2.rectangle(frame,(160,80),(480,400),(r,g,b),2)
    cv2.imshow('Face or No Face',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # 
