# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 21:48:41 2020

@author: sarath
"""


import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential #for initializing
from tensorflow.keras.layers import Dense  #adding layers
from tensorflow.keras.layers import Conv2D  #adding convolution layer
from tensorflow.keras.layers import MaxPooling2D  #max pooling
from tensorflow.keras.layers import Flatten,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

emoji = cv2.imread('./emojis/neutral.png')
cartoon=cv2.imread('./emojis/neutral.png')

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

gender_model = Sequential()
gender_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96,1)))
gender_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
gender_model.add(MaxPooling2D(pool_size=(2, 2)))
gender_model.add(Dropout(0.25))
gender_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gender_model.add(MaxPooling2D(pool_size=(2, 2)))
gender_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
gender_model.add(MaxPooling2D(pool_size=(2, 2)))
gender_model.add(Dropout(0.25))
gender_model.add(Flatten())
gender_model.add(Dense(1024, activation='relu'))
gender_model.add(Dropout(0.5))
gender_model.add(Dense(2, activation='softmax'))
gender_model.load_weights('gender_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ",1: "Contempt", 2: "Disgusted", 3: "  Fearful  ", 4: "   Happy   ", 5: "  Neutral  ", 6: "    Sad    ", 7: "  Surprised  "}


emoji_dist={0:"./emojis/angry.png",1:"./emojis/wangry.png",2:"./emojis/disgusted.png",3:"./emojis/fearful.png",4:"./emojis/happy.png",5:"./emojis/neutral.png",6:"./emojis/sad.png",7:"./emojis/surpriced.png"}


faceCascade=cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
genderList=['Male','Female']
maskList=['Mask','No Mask']
# model = load_model('gender.h5')
maskmodel = load_model("facemask.h5")
maskNet = load_model("mask_detector.model")
#img=cv2.imread('img3.jpg')
cap=cv2.VideoCapture(0)
cap.set(3,640)  #width
cap.set(4,480)  #height
cap.set(10,100)  #brightness
show_text=[0]
num_down = 2       # number of downsampling steps
num_bilateral = 7  # number of bilateral filtering steps
while True:
    success,img=cap.read()
    #cv2.imshow('Video',img)
    img=cv2.flip(img,1)
    ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(ig,1.1,2)
    img_color=img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9,  sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    mfaces=[]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray_frame = ig[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        show_text[0]=maxindex
        emoji=cv2.imread(emoji_dist[maxindex])
        face = ig[y:y + h, x:x + w].copy()
        xface = img[y:y + h, x:x + w].copy()
        # face_crop = cv2.resize(face, (96,96))
        # face_crop=np.expand_dims(face_crop,axis=-1)
        # face_crop=np.expand_dims(face_crop,axis=0)
        # if(np.max(face_crop)>1):
        #     face_crop=face_crop/255.0
        # gen=model.predict_classes(face_crop)
        # gender = genderList[gen[0][0]]
        # gface= ig[y:y + h, x:x + w].copy()
        gen_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (96, 96)), -1), 0)
        genpred = gender_model.predict(gen_img)
        genindex = int(np.argmax(genpred))
        gender = genderList[genindex]

        # face = cv2.cvtColor(xface, cv2.COLOR_BGR2RGB)
        # face = cv2.resize(face, (224, 224))
        # face = img_to_array(face)
        # face = preprocess_input(face)
        # mfaces.append(face)
        # mface = np.array(mfaces, dtype="float32")
        # pred = maskNet.predict(mface, batch_size=32)
        # (mask, withoutMask) = pred[0]
        # label = "Mask" if mask > withoutMask else "No Mask"
        mface = cv2.cvtColor(xface, cv2.COLOR_BGR2RGB)
        mface = cv2.resize(mface, (64,64))
        # mface = np.expand_dims(mface,axis=-1)
        mface = np.expand_dims(mface,axis=0)
        if(np.max(mface)>1):
            mface=mface/255.0
        mout=maskmodel.predict_classes(mface)
        mask = maskList[mout[0][0]]

        cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'{gender}, {mask}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    #cv2.putText(img,show_text[0],(300,200),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3)
    cv2.imshow('Result',img)
    cv2.imshow('Emoji',emoji)
    cv2.imshow('Cartoon',img_cartoon)
    if cv2.waitKey(1) & 0xFF==ord('q'):
       break
