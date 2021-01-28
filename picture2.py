# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 22:09:02 2020

@author: sarath
"""

import cv2
import numpy as np
import pandas as pd
import sys

from tensorflow.keras.models import Sequential #for initializing
from tensorflow.keras.layers import Dense  #adding layers
from tensorflow.keras.layers import Conv2D  #adding convolution layer
from tensorflow.keras.layers import MaxPooling2D  #max pooling
from tensorflow.keras.layers import Flatten,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


#emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
emoji_dist_male={0:"./emojis/mangry.png",1:"./emojis/mdisgusted.png",2:"./emojis/mfearful.png",3:"./emojis/mhappy.png",4:"./emojis/mneutral.png",5:"./emojis/msad.png",6:"./emojis/msurpriced.png"}
emoji_dist_female={0:"./emojis/fangry.png",1:"./emojis/fdisgusted.png",2:"./emojis/ffearful.png",3:"./emojis/fhappy.png",4:"./emojis/fneutral.png",5:"./emojis/fsad.png",6:"./emojis/fsurpriced.png"}

faceCascade=cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(35-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)
# model = load_model('gender_detection.model')
emoji = cv2.imread('./emojis/loading.png')
cartoon=cv2.imread('./emojis/loading.png')
# model = load_model('gender.h5')
maskmodel = load_model("facemask_model.h5")
maskList=['Mask','No Mask']

class PhotoCamera(object):
    def __init__(self):
        self.emotion_model = Sequential()

        self.emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        self.emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))

        self.emotion_model.add(Flatten())
        self.emotion_model.add(Dense(1024, activation='relu'))
        self.emotion_model.add(Dropout(0.5))
        self.emotion_model.add(Dense(7, activation='softmax'))
        self.emotion_model.load_weights('emotion_model.h5')



    def get_pframe(self,file_path):
        print(file_path)
        # img = image.load_img(file_path, target_size=(48, 48))
        # cv2.ocl.setUseOpenCL(False)
        # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        img=cv2.imread(file_path)
        # img=resize(img,(48,48))
        ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(ig,1.3,5)

        for (x,y,w,h) in faces:
            int_averages=np.empty((3))
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            gray_frame = ig[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
            prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            face = img[y:y + h, x:x + w].copy()
            gface= ig[y:y + h, x:x + w].copy()
            # noseimg=noseCascade.detectMultiScale(ig,1.1,2)
            # i=1
            # for (a, b, c, d) in noseimg:
            #     cv2.rectangle(img, (a,b), (a+c, b+d), (255,255,255), 2)
            #     cv2.putText(img, str(i) , (a+10, b-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            #     i+=1
            # gblob = cv2.dnn.blobFromImage(face, 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=True)
            # genderNet.setInput(gblob)
            # genderPreds=genderNet.forward()
            # gender=genderList[genderPreds[0].argmax()]
            # global emoji
            # if gender=='Male':
            #     emoji=cv2.imread(emoji_dist_male[maxindex])
            # elif gender=='Female':
            #     emoji=cv2.imread(emoji_dist_female[maxindex])

            # ablob = cv2.dnn.blobFromImage(face, 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # ageNet.setInput(ablob)
            # agePreds=ageNet.forward()
            # age=ageList[agePreds[0].argmax()]

            # face=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # height, width, _ = np.shape(face)
            # # calculate the average color of each row of our image
            # avg_color_per_row = np.average(img, axis=0)
            #
            # # calculate the averages of our rows
            # avg_colors = np.average(avg_color_per_row, axis=0)
            #
            # # avg_color is a tuple in BGR order of the average colors
            # # but as float values
            # # print(f'avg_colors: {avg_colors}')
            #
            # # so, convert that array to integers
            # int_averages = np.array(avg_colors, dtype=np.uint8)
            # print(int_averages)
            # if int_averages[0]<=90 and int_averages[1]<=60 and int_averages[2]<=60:
            #     skin = 'Dark'
            # elif int_averages[0]<=100 and int_averages[1]<=120 and int_averages[2]<=150:
            #     skin = 'Medium'
            # else:
            #     skin = 'Fair'

            # print(f'int_averages: {int_averages}')

            # create a new image of the same height/width as the original
            # average_image = np.zeros((height, width, 3), np.uint8)
            # and fill its pixels with our average color
            # average_image[:] = int_averages

            mface = cv2.resize(gface, (112,112))
            mface = np.expand_dims(mface,axis=-1)
            mface = np.expand_dims(mface,axis=0)
            if(np.max(mface)>1):
                mface=mface/255.0
            mout=maskmodel.predict_classes(mface)
            mask = maskList[mout[0][0]]

            cv2.putText(img, f'{emotion_dict[maxindex]}, {mask}', (x+10, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(img, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            # cv2.putText(img, f'{skin}, {int_averages}', (x+10, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            # int_averages=np.empty((3))

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def  get_pemoji(self):
        global emoji
        ret, emj = cv2.imencode('.jpg', emoji)
        return emj.tobytes()

    def get_pcartoon(self,file_path):
        img=cv2.imread(file_path)
        ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        global cartoon
        cartoon=img
        img_color= img
        num_down = 2
        num_bilateral = 7
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9,  sigmaSpace=7)
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(ig, 7)
        img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        try:
            img_cart = cv2.bitwise_and(img_color, img_edge)
            img_cartoon = np.vstack((img_cart,img_edge))

        except:
            # img_blur = cv2.medianBlur(img_color, 7)
            # img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
            # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            img_cartoon = img_edge
        cartoon=img_cartoon
        ret, car = cv2.imencode('.jpg', cartoon)
        return car.tobytes()
