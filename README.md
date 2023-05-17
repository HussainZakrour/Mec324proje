# Mec324proje
from google.colab import drive
drive.mount('/content/drive')


import tensorflow as tf
import numpy as np
import pandas as pd
import random

from tensorflow import keras
from keras import layers
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import cv2
from cv2 import imread
from google.colab.patches import cv2_imshow
from google.colab import files
from cv2 import rectangle

face_cascade= cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'))
pixels = cv2.imread('/content/drive/MyDrive/projemec324/test/WhatsApp Image 2023-05-14 at 19.12.30.jpeg')
bboxes = face_cascade.detectMultiScale(pixels)
for box in bboxes:
    print(box) 
    
for box in bboxes:
    
    x, y, width, height = box
    x2, y2 = x + width, y + height
    
    rectangle(pixels, (x, y), (x2, y2), (0,0,255), 4)
   
cv2_imshow(pixels),
#display(pixels),
print( 'number of detected faces :' ,len(bboxes))


def resize(img):
   print('Original Dimensions : ',img.shape)
   scale_percent = 60 
   width = int(img.shape[1] * scale_percent / 300)
   height = int(img.shape[0] * scale_percent / 300)
   dim = (width, height)
   
   resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
   print('Resized Dimensions : ',resized.shape)
    
   #cv2_imshow(resized)
   
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()
   return resized

