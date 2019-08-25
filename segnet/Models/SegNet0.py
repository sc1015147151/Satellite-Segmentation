#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np 
from keras import *
from keras.models import Sequential  
from keras.layers import *
from keras.layers import Input
from keras.utils.np_utils import to_categorical  
from keras.preprocessing.image import img_to_array  
from keras.callbacks import ModelCheckpoint ,TensorBoard

from utils import *
from sklearn.preprocessing import LabelEncoder  
from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
from keras import backend as K 
from keras.applications import vgg16
from keras.layers import Input
def SegNet0(): 
  n_label=2
  img_w=256
  img_h=256
  model = Sequential()  
  #encoder  
  model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,4),padding='same',activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))  
  model.add(BatchNormalization())  
  model.add(MaxPooling2D(pool_size=(2,2)))  
  #(128,128)  
  model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(MaxPooling2D(pool_size=(2, 2)))  
  #(64,64)  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(MaxPooling2D(pool_size=(2, 2)))  
  #(32,32)  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(MaxPooling2D(pool_size=(2, 2)))  
  #(16,16)  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(MaxPooling2D(pool_size=(2, 2)))  
  #(8,8)  
  #decoder  
  model.add(UpSampling2D(size=(2,2)))  
  #(16,16)  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(UpSampling2D(size=(2, 2)))  
  #(32,32)  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(UpSampling2D(size=(2, 2)))  
  #(64,64)  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(UpSampling2D(size=(2, 2)))  
  #(128,128)  
  model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(UpSampling2D(size=(2, 2)))  
  #(256,256)  
  model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(4,img_w, img_h), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))  
  model.add(BatchNormalization())  
  model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))  
  model.add(Reshape((n_label,img_w*img_h)))  
  #axis=1和axis=2互换位置，等同于np.swapaxes(layer,1,2)  
  model.add(Permute((2,1)))  
  model.add(Activation('softmax'))  
  model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
  
  return model  
if __name__ == '__main__':
    m =  SegNet0()
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_SegNet0.png')
    print(len(m.layers))
    m.summary()
