# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 22:09:02 2020

@author: sarath
"""

import cv2
import numpy as np
import pandas as pd
import sys
import random

from tensorflow.keras.models import Sequential #for initializing
from tensorflow.keras.layers import Dense  #adding layers
from tensorflow.keras.layers import Conv2D  #adding convolution layer
from tensorflow.keras.layers import MaxPooling2D  #max pooling
from tensorflow.keras.layers import Flatten,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


#emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
emoji_dist_male={0:"./emojis/mangry.png",1:"./emojis/mdisgusted.png",2:"./emojis/mfearful.png",3:"./emojis/mhappy.png",4:"./emojis/mneutral.png",5:"./emojis/msad.png",6:"./emojis/msurprised.png"}
emoji_dist_male11={0:"./emojis/set1/male/MA11/mangry.png",1:"./emojis/set1/male/MA11/mdisgusted.png",2:"./emojis/set1/male/MA11/mfearful.png",3:"./emojis/set1/male/MA11/mhappy.png",4:"./emojis/set1/male/MA11/mneutral.png",5:"./emojis/set1/male/MA11/msad.png",6:"./emojis/set1/male/MA11/msurprised.png"}
emoji_dist_male12={0:"./emojis/set1/male/MA12/mangry.png",1:"./emojis/set1/male/MA12/mdisgusted.png",2:"./emojis/set1/male/MA12/mfearful.png",3:"./emojis/set1/male/MA12/mhappy.png",4:"./emojis/set1/male/MA12/mneutral.png",5:"./emojis/set1/male/MA12/msad.png",6:"./emojis/set1/male/MA12/msurprised.png"}
emoji_dist_male13={0:"./emojis/set1/male/MA13/mangry.png",1:"./emojis/set1/male/MA13/mdisgusted.png",2:"./emojis/set1/male/MA13/mfearful.png",3:"./emojis/set1/male/MA13/mhappy.png",4:"./emojis/set1/male/MA13/mneutral.png",5:"./emojis/set1/male/MA13/msad.png",6:"./emojis/set1/male/MA13/msurprised.png"}
emoji_dist_male14={0:"./emojis/set1/male/MA14/mangry.png",1:"./emojis/set1/male/MA14/mdisgusted.png",2:"./emojis/set1/male/MA14/mfearful.png",3:"./emojis/set1/male/MA14/mhappy.png",4:"./emojis/set1/male/MA14/mneutral.png",5:"./emojis/set1/male/MA14/msad.png",6:"./emojis/set1/male/MA14/msurprised.png"}
emoji_dist_male15={0:"./emojis/set1/male/MA15/mangry.png",1:"./emojis/set1/male/MA15/mdisgusted.png",2:"./emojis/set1/male/MA15/mfearful.png",3:"./emojis/set1/male/MA15/mhappy.png",4:"./emojis/set1/male/MA15/mneutral.png",5:"./emojis/set1/male/MA15/msad.png",6:"./emojis/set1/male/MA15/msurprised.png"}
emoji_dist_male16={0:"./emojis/set1/male/MA16/mangry.png",1:"./emojis/set1/male/MA16/mdisgusted.png",2:"./emojis/set1/male/MA16/mfearful.png",3:"./emojis/set1/male/MA16/mhappy.png",4:"./emojis/set1/male/MA16/mneutral.png",5:"./emojis/set1/male/MA16/msad.png",6:"./emojis/set1/male/MA16/msurprised.png"}
emoji_dist_male17={0:"./emojis/set1/male/MA17/mangry.png",1:"./emojis/set1/male/MA17/mdisgusted.png",2:"./emojis/set1/male/MA17/mfearful.png",3:"./emojis/set1/male/MA17/mhappy.png",4:"./emojis/set1/male/MA17/mneutral.png",5:"./emojis/set1/male/MA17/msad.png",6:"./emojis/set1/male/MA17/msurprised.png"}

emoji_dist_male21={0:"./emojis/set2/male/MA21/mangry.png",1:"./emojis/set2/male/MA21/mdisgusted.png",2:"./emojis/set2/male/MA21/mfearful.png",3:"./emojis/set2/male/MA21/mhappy.png",4:"./emojis/set2/male/MA21/mneutral.png",5:"./emojis/set2/male/MA21/msad.png",6:"./emojis/set2/male/MA21/msurprised.png"}
emoji_dist_male22={0:"./emojis/set2/male/MA22/mangry.png",1:"./emojis/set2/male/MA22/mdisgusted.png",2:"./emojis/set2/male/MA22/mfearful.png",3:"./emojis/set2/male/MA22/mhappy.png",4:"./emojis/set2/male/MA22/mneutral.png",5:"./emojis/set2/male/MA22/msad.png",6:"./emojis/set2/male/MA22/msurprised.png"}
emoji_dist_male23={0:"./emojis/set2/male/MA23/mangry.png",1:"./emojis/set2/male/MA23/mdisgusted.png",2:"./emojis/set2/male/MA23/mfearful.png",3:"./emojis/set2/male/MA23/mhappy.png",4:"./emojis/set2/male/MA23/mneutral.png",5:"./emojis/set2/male/MA23/msad.png",6:"./emojis/set2/male/MA23/msurprised.png"}
emoji_dist_male24={0:"./emojis/set2/male/MA24/mangry.png",1:"./emojis/set2/male/MA24/mdisgusted.png",2:"./emojis/set2/male/MA24/mfearful.png",3:"./emojis/set2/male/MA24/mhappy.png",4:"./emojis/set2/male/MA24/mneutral.png",5:"./emojis/set2/male/MA24/msad.png",6:"./emojis/set2/male/MA24/msurprised.png"}
emoji_dist_male25={0:"./emojis/set2/male/MA25/mangry.png",1:"./emojis/set2/male/MA25/mdisgusted.png",2:"./emojis/set2/male/MA25/mfearful.png",3:"./emojis/set2/male/MA25/mhappy.png",4:"./emojis/set2/male/MA25/mneutral.png",5:"./emojis/set2/male/MA25/msad.png",6:"./emojis/set2/male/MA25/msurprised.png"}
emoji_dist_male26={0:"./emojis/set2/male/MA26/mangry.png",1:"./emojis/set2/male/MA26/mdisgusted.png",2:"./emojis/set2/male/MA26/mfearful.png",3:"./emojis/set2/male/MA26/mhappy.png",4:"./emojis/set2/male/MA26/mneutral.png",5:"./emojis/set2/male/MA26/msad.png",6:"./emojis/set2/male/MA26/msurprised.png"}
emoji_dist_male27={0:"./emojis/set2/male/MA27/mangry.png",1:"./emojis/set2/male/MA27/mdisgusted.png",2:"./emojis/set2/male/MA27/mfearful.png",3:"./emojis/set2/male/MA27/mhappy.png",4:"./emojis/set2/male/MA27/mneutral.png",5:"./emojis/set2/male/MA27/msad.png",6:"./emojis/set2/male/MA27/msurprised.png"}



emoji_dist_female={0:"./emojis/fangry.png",1:"./emojis/fdisgusted.png",2:"./emojis/ffearful.png",3:"./emojis/fhappy.png",4:"./emojis/fneutral.png",5:"./emojis/fsad.png",6:"./emojis/fsurprised.png"}
emoji_dist_female11={0:"./emojis/set1/female/FA11/fangry.png",1:"./emojis/set1/female/FA11/fdisgusted.png",2:"./emojis/set1/female/FA11/ffearful.png",3:"./emojis/set1/female/FA11/fhappy.png",4:"./emojis/set1/female/FA11/fneutral.png",5:"./emojis/set1/female/FA11/fsad.png",6:"./emojis/set1/female/FA11/fsurprised.png"}
emoji_dist_female12={0:"./emojis/set1/female/FA12/fangry.png",1:"./emojis/set1/female/FA12/fdisgusted.png",2:"./emojis/set1/female/FA12/ffearful.png",3:"./emojis/set1/female/FA12/fhappy.png",4:"./emojis/set1/female/FA12/fneutral.png",5:"./emojis/set1/female/FA12/fsad.png",6:"./emojis/set1/female/FA12/fsurprised.png"}
emoji_dist_female13={0:"./emojis/set1/female/FA13/fangry.png",1:"./emojis/set1/female/FA13/fdisgusted.png",2:"./emojis/set1/female/FA13/ffearful.png",3:"./emojis/set1/female/FA13/fhappy.png",4:"./emojis/set1/female/FA13/fneutral.png",5:"./emojis/set1/female/FA13/fsad.png",6:"./emojis/set1/female/FA13/fsurprised.png"}
emoji_dist_female14={0:"./emojis/set1/female/FA14/fangry.png",1:"./emojis/set1/female/FA14/fdisgusted.png",2:"./emojis/set1/female/FA14/ffearful.png",3:"./emojis/set1/female/FA14/fhappy.png",4:"./emojis/set1/female/FA14/fneutral.png",5:"./emojis/set1/female/FA14/fsad.png",6:"./emojis/set1/female/FA14/fsurprised.png"}
emoji_dist_female15={0:"./emojis/set1/female/FA15/fangry.png",1:"./emojis/set1/female/FA15/fdisgusted.png",2:"./emojis/set1/female/FA15/ffearful.png",3:"./emojis/set1/female/FA15/fhappy.png",4:"./emojis/set1/female/FA15/fneutral.png",5:"./emojis/set1/female/FA15/fsad.png",6:"./emojis/set1/female/FA15/fsurprised.png"}
emoji_dist_female16={0:"./emojis/set1/female/FA16/fangry.png",1:"./emojis/set1/female/FA16/fdisgusted.png",2:"./emojis/set1/female/FA16/ffearful.png",3:"./emojis/set1/female/FA16/fhappy.png",4:"./emojis/set1/female/FA16/fneutral.png",5:"./emojis/set1/female/FA16/fsad.png",6:"./emojis/set1/female/FA16/fsurprised.png"}
emoji_dist_female17={0:"./emojis/set1/female/FA17/fangry.png",1:"./emojis/set1/female/FA17/fdisgusted.png",2:"./emojis/set1/female/FA17/ffearful.png",3:"./emojis/set1/female/FA17/fhappy.png",4:"./emojis/set1/female/FA17/fneutral.png",5:"./emojis/set1/female/FA17/fsad.png",6:"./emojis/set1/female/FA17/fsurprised.png"}

emoji_dist_female21={0:"./emojis/set2/female/FA21/fangry.png",1:"./emojis/set2/female/FA21/fdisgusted.png",2:"./emojis/set2/female/FA21/ffearful.png",3:"./emojis/set2/female/FA21/fhappy.png",4:"./emojis/set2/female/FA21/fneutral.png",5:"./emojis/set2/female/FA21/fsad.png",6:"./emojis/set2/female/FA21/fsurprised.png"}
emoji_dist_female22={0:"./emojis/set2/female/FA22/fangry.png",1:"./emojis/set2/female/FA22/fdisgusted.png",2:"./emojis/set2/female/FA22/ffearful.png",3:"./emojis/set2/female/FA22/fhappy.png",4:"./emojis/set2/female/FA22/fneutral.png",5:"./emojis/set2/female/FA22/fsad.png",6:"./emojis/set2/female/FA22/fsurprised.png"}
emoji_dist_female23={0:"./emojis/set2/female/FA23/fangry.png",1:"./emojis/set2/female/FA23/fdisgusted.png",2:"./emojis/set2/female/FA23/ffearful.png",3:"./emojis/set2/female/FA23/fhappy.png",4:"./emojis/set2/female/FA23/fneutral.png",5:"./emojis/set2/female/FA23/fsad.png",6:"./emojis/set2/female/FA23/fsurprised.png"}
emoji_dist_female24={0:"./emojis/set2/female/FA24/fangry.png",1:"./emojis/set2/female/FA24/fdisgusted.png",2:"./emojis/set2/female/FA24/ffearful.png",3:"./emojis/set2/female/FA24/fhappy.png",4:"./emojis/set2/female/FA24/fneutral.png",5:"./emojis/set2/female/FA24/fsad.png",6:"./emojis/set2/female/FA24/fsurprised.png"}
emoji_dist_female25={0:"./emojis/set2/female/FA25/fangry.png",1:"./emojis/set2/female/FA25/fdisgusted.png",2:"./emojis/set2/female/FA25/ffearful.png",3:"./emojis/set2/female/FA25/fhappy.png",4:"./emojis/set2/female/FA25/fneutral.png",5:"./emojis/set2/female/FA25/fsad.png",6:"./emojis/set2/female/FA25/fsurprised.png"}
emoji_dist_female26={0:"./emojis/set2/female/FA26/fangry.png",1:"./emojis/set2/female/FA26/fdisgusted.png",2:"./emojis/set2/female/FA26/ffearful.png",3:"./emojis/set2/female/FA26/fhappy.png",4:"./emojis/set2/female/FA26/fneutral.png",5:"./emojis/set2/female/FA26/fsad.png",6:"./emojis/set2/female/FA26/fsurprised.png"}
emoji_dist_female27={0:"./emojis/set2/female/FA27/fangry.png",1:"./emojis/set2/female/FA27/fdisgusted.png",2:"./emojis/set2/female/FA27/ffearful.png",3:"./emojis/set2/female/FA27/fhappy.png",4:"./emojis/set2/female/FA27/fneutral.png",5:"./emojis/set2/female/FA27/fsad.png",6:"./emojis/set2/female/FA27/fsurprised.png"}



faceCascade=cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(35-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
maskList=['Mask','No Mask']
glassList=['Glasses','No Glasses']

ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)
emoji = cv2.imread('./emojis/loading.png')
cartoon=cv2.imread('./emojis/loading.png')
num_down = 2
num_bilateral = 7
rn=0

# gender model
gendermodel = load_model("gender_model.h5")

# load the face mask detector model from disk
maskmodel = load_model("facemask.h5")
# maskNet = load_model("mask_detector.model")

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

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
        global rn
        rn=random.randint(1,2)

    def __del__(self):
        self.cap.release()


    def get_frame(self):
        success, img = self.cap.read()
        cv2.ocl.setUseOpenCL(False)

        while True:
            success,img=self.cap.read()
            img=cv2.flip(img,1)
            ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(ig,1.3,5)
            global cartoon
            cartoon=img
            img_color= cartoon
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
            img_cartoon = cv2.bitwise_and(img_color, img_edge)
            cartoon=img_cartoon

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                face = img[y:y + h, x:x + w].copy()
                gface= ig[y:y + h, x:x + w].copy()

                face_crop = cv2.resize(gface, (96,96))
                face_crop=np.expand_dims(face_crop,axis=-1)
                face_crop=np.expand_dims(face_crop,axis=0)
                if(np.max(face_crop)>1):
                    face_crop=face_crop/255.0
                gen=gendermodel.predict_classes(face_crop)
                gender = genderList[gen[0][0]]

                ablob = cv2.dnn.blobFromImage(face, 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(ablob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]

                mface = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                mface = cv2.resize(mface, (64,64))
                # mface = np.expand_dims(mface,axis=-1)
                mface = np.expand_dims(mface,axis=0)
                if(np.max(mface)>1):
                    mface=mface/255.0
                mout=maskmodel.predict_classes(mface)
                mask = maskList[mout[0][0]]

                if mask == 'No Mask':
                    roi_gray_frame = ig[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    prediction = self.emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))

                    global emoji
                    global rn
                    if rn==1:
                        if gender=='Male':
                            if age=='(0-2)':
                                emoji=cv2.imread(emoji_dist_male11[maxindex])
                            elif age=='(4-6)':
                                emoji=cv2.imread(emoji_dist_male11[maxindex])
                            elif age=='(8-12)':
                                emoji=cv2.imread(emoji_dist_male12[maxindex])
                            elif age=='(15-20)':
                                emoji=cv2.imread(emoji_dist_male13[maxindex])
                            elif age=='(21-32)':
                                emoji=cv2.imread(emoji_dist_male14[maxindex])
                            elif age=='(35-43)':
                                emoji=cv2.imread(emoji_dist_male15[maxindex])
                            elif age=='(48-53)':
                                emoji=cv2.imread(emoji_dist_male16[maxindex])
                            else:
                                emoji=cv2.imread(emoji_dist_male17[maxindex])
                        elif gender=='Female':
                            if age=='(0-2)':
                                emoji=cv2.imread(emoji_dist_female11[maxindex])
                            elif age=='(4-6)':
                                emoji=cv2.imread(emoji_dist_female11[maxindex])
                            elif age=='(8-12)':
                                emoji=cv2.imread(emoji_dist_female12[maxindex])
                            elif age=='(15-20)':
                                emoji=cv2.imread(emoji_dist_female13[maxindex])
                            elif age=='(21-32)':
                                emoji=cv2.imread(emoji_dist_female14[maxindex])
                            elif age=='(35-43)':
                                emoji=cv2.imread(emoji_dist_female15[maxindex])
                            elif age=='(48-53)':
                                emoji=cv2.imread(emoji_dist_female16[maxindex])
                            else:
                                emoji=cv2.imread(emoji_dist_female17[maxindex])
                    else:
                        if gender=='Male':
                            if age=='(0-2)':
                                emoji=cv2.imread(emoji_dist_male21[maxindex])
                            elif age=='(4-6)':
                                emoji=cv2.imread(emoji_dist_male22[maxindex])
                            elif age=='(8-12)':
                                emoji=cv2.imread(emoji_dist_male23[maxindex])
                            elif age=='(15-20)':
                                emoji=cv2.imread(emoji_dist_male24[maxindex])
                            elif age=='(21-32)':
                                emoji=cv2.imread(emoji_dist_male25[maxindex])
                            elif age=='(35-43)':
                                emoji=cv2.imread(emoji_dist_male26[maxindex])
                            elif age=='(48-53)':
                                emoji=cv2.imread(emoji_dist_male27[maxindex])
                            else:
                                emoji=cv2.imread(emoji_dist_male27[maxindex])
                        elif gender=='Female':
                            if age=='(0-2)':
                                emoji=cv2.imread(emoji_dist_female21[maxindex])
                            elif age=='(4-6)':
                                emoji=cv2.imread(emoji_dist_female22[maxindex])
                            elif age=='(8-12)':
                                emoji=cv2.imread(emoji_dist_female23[maxindex])
                            elif age=='(15-20)':
                                emoji=cv2.imread(emoji_dist_female24[maxindex])
                            elif age=='(21-32)':
                                emoji=cv2.imread(emoji_dist_female25[maxindex])
                            elif age=='(35-43)':
                                emoji=cv2.imread(emoji_dist_female26[maxindex])
                            elif age=='(48-53)':
                                emoji=cv2.imread(emoji_dist_female27[maxindex])
                            else:
                                emoji=cv2.imread(emoji_dist_female27[maxindex])

                    cv2.putText(img, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, f'{gender}, {age}, {mask}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

                else:
                    if gender=='Male':
                        emoji=cv2.imread('./emojis/mmask.png')
                    else:
                        emoji=cv2.imread('./emojis/fmask.png')

                    cv2.putText(img, 'Neutral', (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, f'{gender}, {age}, {mask}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

    def  get_emoji(self):
        global emoji
        ret, emj = cv2.imencode('.jpg', emoji)
        return emj.tobytes()

    def get_cartoon(self):
        global cartoon
        ret, car = cv2.imencode('.jpg', cartoon)
        return car.tobytes()
