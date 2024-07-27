import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("models\keras_model.h5","models\labels.txt")

offset=20
img_size=300
counter=0

labels=["A","B","C","D","E","F"]

while True:
    success, img= cap.read()
    img_output=img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand=hands[0]
        x, y, w, h = hand["bbox"]

        img_white=np.ones((img_size,img_size,3),np.uint8)*255

        img_crop=img[y-offset: y + h+offset, x-offset: x + w+offset]

        img_crop_shape=img_crop.shape

        aspect_ratio= h/w
        if aspect_ratio>1:
            k=img_size/h
            w_cal=math.ceil(k*w)
            img_resize=cv2.resize(img_crop,(w_cal,img_size))
            img_resize_shape=img_resize.shape
            w_gap=math.ceil((img_size-w_cal)/2)
            img_white[:,w_gap:w_cal+w_gap]=img_resize

            print(prediction,index)
            prediction, index=classifier.getPrediction(img_white,draw=False)

        else:
            k=img_size/w
            h_cal=math.ceil(k*h)
            img_resize=cv2.resize(img_crop,(img_size,h_cal))
            img_resize_shape=img_resize.shape
            h_gap=math.ceil((img_size-h_cal)/2)
            img_white[h_gap:h_cal+h_gap,:]=img_resize

            prediction, index=classifier.getPrediction(img_white,draw=False)
            # print(index)
            # print(prediction,index)

        cv2.rectangle(img_output,(x-offset,y-offset),
                      (x-offset+150,y-offset),(0,255,0),cv2.FILLED)
        cv2.putText(img_output, labels[index], (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 3)
        cv2.rectangle(img_output,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,255,0),4)

        # cv2.imshow("image crop",img_crop)
        # cv2.imshow("image white",img_white)



    cv2.imshow("image",img_output)
    cv2.waitKey(1)
