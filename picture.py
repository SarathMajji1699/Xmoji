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
from tensorflow.keras.layers import Flatten,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


#emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
emoji_dist_male={0:"./emojis/mangry.png",1:"./emojis/mdisgusted.png",2:"./emojis/mfearful.png",3:"./emojis/mhappy.png",4:"./emojis/mneutral.png",5:"./emojis/msad.png",6:"./emojis/msurpriced.png"}
emoji_dist_female={0:"./emojis/fangry.png",1:"./emojis/fdisgusted.png",2:"./emojis/ffearful.png",3:"./emojis/fhappy.png",4:"./emojis/fneutral.png",5:"./emojis/fsad.png",6:"./emojis/fsurpriced.png"}

faceCascade=cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"


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

prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

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
        count=0
        for (x,y,w,h) in faces:
            count+=1
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            gray_frame = ig[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
            prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

            face = img[y:y + h, x:x + w].copy()
            gface= ig[y:y + h, x:x + w].copy()

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
            print(emotion_dict[maxindex],mask)

        if count==0:
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104.0, 177.0, 123.0))
            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()
            for i in range(0, detections.shape[2]):
        		# extract the confidence (i.e., probability) associated with
        		# the detection
                confidence = detections[0, 0, i, 2]

        		# filter out weak detections by ensuring the confidence is
        		# greater than the minimum confidence
                if confidence > 0.5:
        			# compute the (x, y)-coordinates of the bounding box for
        			# the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

        			# ensure the bounding boxes fall within the dimensions of
        			# the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        			# extract the face ROI, convert it from BGR to RGB channel
        			# ordering, resize it to 224x224, and preprocess it
                    face = img[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    cv2.rectangle(img,(startX,startY),(endX,endY),(255,0,0),2)
                    # mface = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    gface = ig[startY:endY, startX:endX]
                    mface = cv2.resize(gface, (112,112))
                    mface = np.expand_dims(mface,axis=-1)
                    mface = np.expand_dims(mface,axis=0)
                    if(np.max(mface)>1):
                        mface=mface/255.0
                    mout=maskmodel.predict_classes(mface)
                    mask = maskList[mout[0][0]]

                    gray_frame = ig[startY:endY, startX:endX]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
                    prediction = self.emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction,axis=-1))

                    cv2.putText(img, f'{emotion_dict[maxindex]}, {mask}', (startX+10, startY-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print(emotion_dict[maxindex],mask)

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






















# import cv2
# import numpy as np
# import pandas as pd
# import sys
#
# from tensorflow.keras.models import Sequential #for initializing
# from tensorflow.keras.layers import Dense  #adding layers
# from tensorflow.keras.layers import Conv2D  #adding convolution layer
# from tensorflow.keras.layers import MaxPooling2D  #max pooling
# from tensorflow.keras.layers import Flatten,Dropout,Flatten
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing import image
# # from skimage.transform import resize
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
#
#
# emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
#
#
# #emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
# emoji_dist_male={0:"./emojis/mangry.png",1:"./emojis/mdisgusted.png",2:"./emojis/mfearful.png",3:"./emojis/mhappy.png",4:"./emojis/mneutral.png",5:"./emojis/msad.png",6:"./emojis/msurpriced.png"}
# emoji_dist_female={0:"./emojis/fangry.png",1:"./emojis/fdisgusted.png",2:"./emojis/ffearful.png",3:"./emojis/fhappy.png",4:"./emojis/fneutral.png",5:"./emojis/fsad.png",6:"./emojis/fsurpriced.png"}
#
# faceCascade=cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
#
# prototxtPath = r"deploy.prototxt"
# weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#
#
# MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# # ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-32)', '(35-43)', '(48-53)', '(60-100)']
# genderList=['Male','Female']
#
#
# model = load_model('gender_model.h5')
# emoji = cv2.imread('./emojis/loading.png')
# cartoon=cv2.imread('./emojis/loading.png')
# # model = load_model('gender.h5')
# # maskmodel = load_model("facemask.h5")
# maskmodel = load_model("facemask_model.h5")
# maskList=['Mask','No Mask']
#
# class PhotoCamera(object):
#     def __init__(self):
#         self.emotion_model = Sequential()
#
#         self.emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
#         self.emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#         self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.emotion_model.add(Dropout(0.25))
#
#         self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#         self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
#         self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.emotion_model.add(Dropout(0.25))
#
#         self.emotion_model.add(Flatten())
#         self.emotion_model.add(Dense(1024, activation='relu'))
#         self.emotion_model.add(Dropout(0.5))
#         self.emotion_model.add(Dense(7, activation='softmax'))
#         self.emotion_model.load_weights('emotion_model.h5')
#
#
#
#     def get_pframe(self,file_path):
#         print(file_path)
#         img=cv2.imread(file_path)
#         # img=resize(img,(48,48))
#         ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         # faces=faceCascade.detectMultiScale(ig,1.3,5)
#         (h, w) = img.shape[:2]
#         blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104.0, 177.0, 123.0))
#         # pass the blob through the network and obtain the face detections
#         faceNet.setInput(blob)
#         detections = faceNet.forward()
#         for i in range(0, detections.shape[2]):
#     		# extract the confidence (i.e., probability) associated with
#     		# the detection
#             confidence = detections[0, 0, i, 2]
#
#     		# filter out weak detections by ensuring the confidence is
#     		# greater than the minimum confidence
#             if confidence > 0.5:
#     			# compute the (x, y)-coordinates of the bounding box for
#     			# the object
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#
#     			# ensure the bounding boxes fall within the dimensions of
#     			# the frame
#                 (startX, startY) = (max(0, startX), max(0, startY))
#                 (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
#
#     			# extract the face ROI, convert it from BGR to RGB channel
#     			# ordering, resize it to 224x224, and preprocess it
#                 face = img[startY:endY, startX:endX]
#                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 face = cv2.resize(face, (224, 224))
#                 face = img_to_array(face)
#                 face = preprocess_input(face)
#                 cv2.rectangle(img,(startX,startY),(endX,endY),(255,0,0),2)
#                 # mface = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#                 gface = ig[startY:endY, startX:endX]
#                 mface = cv2.resize(gface, (112,112))
#                 mface = np.expand_dims(mface,axis=-1)
#                 mface = np.expand_dims(mface,axis=0)
#                 if(np.max(mface)>1):
#                     mface=mface/255.0
#                 mout=maskmodel.predict_classes(mface)
#                 mask = maskList[mout[0][0]]
#
#                 gray_frame = ig[startY:endY, startX:endX]
#                 cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
#                 prediction = self.emotion_model.predict(cropped_img)
#                 maxindex = int(np.argmax(prediction,axis=-1))
#
#                 cv2.putText(img, f'{emotion_dict[maxindex]}, {mask}', (startX+10, startY-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#             # cv2.putText(img, f'{gender}, {age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
#             # cv2.putText(img, f'{skin}, {int_averages}', (x+10, y-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
#             # int_averages=np.empty((3))
#
#         ret, jpeg = cv2.imencode('.jpg', img)
#         return jpeg.tobytes()
#
#     def  get_pemoji(self):
#         global emoji
#         ret, emj = cv2.imencode('.jpg', emoji)
#         return emj.tobytes()
#
#     def get_pcartoon(self,file_path):
#         img=cv2.imread(file_path)
#         ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         global cartoon
#         cartoon=img
#         img_color= img
#         num_down = 2
#         num_bilateral = 7
#         for _ in range(num_down):
#             img_color = cv2.pyrDown(img_color)
#         for _ in range(num_bilateral):
#             img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9,  sigmaSpace=7)
#         for _ in range(num_down):
#             img_color = cv2.pyrUp(img_color)
#         # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img_blur = cv2.medianBlur(ig, 7)
#         img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
#         img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
#         try:
#             img_cart = cv2.bitwise_and(img_color, img_edge)
#             img_cartoon = np.vstack((img_cart,img_edge))
#
#         except:
#             # img_blur = cv2.medianBlur(img_color, 7)
#             # img_edge = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,  blockSize=9, C=2)
#             # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
#             img_cartoon = img_edge
#         cartoon=img_cartoon
#         ret, car = cv2.imencode('.jpg', cartoon)
#         return car.tobytes()
