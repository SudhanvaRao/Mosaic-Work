import tensorflow as tf
from tensorflow import keras
import string
import cv2
import random
import gc
import numpy as np

PATH='/kaggle/input/letterzs/image'
all_symbols=string.ascii_uppercase + string.ascii_lowercase +'0123456789'
index_symbols=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','aa','bb','cc','dd','ee','ff','gg','hh','ii','jj','kk','ll','mm','nn','oo','pp','qq','rr','ss','tt','uu','vv','ww','xx','yy','zz','0','1','2','3','4','5','6','7','8','9']

traindata=[]
for char in index_symbols:
    path=os.path.join(PATH,char)
    for img in os.listdir(path):
        symb_index=index_symbols.index(char)
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        print(os.path.join(path,img))
        image_array=cv2.resize(img_array,(100,100))
        traindata.append([image_array,symb_index])
random.shuffle(traindata)

traindata=[]
for char in index_symbols:
    path=os.path.join(PATH,char)
    for img in os.listdir(path):
        symb_index=index_symbols.index(char)
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        print(os.path.join(path,img))
        image_array=cv2.resize(img_array,(100,100))
        traindata.append([image_array,symb_index])
random.shuffle(traindata)

def model():
    inputs = keras.Input(shape=(100,100,1))
    x = keras.layers.Conv2D(200, (3,3),activation='relu',padding ='same')(inputs)
    x_shortcut = x
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x,x_shortcut])
    x = keras.layers.Conv2D(500, (3,3),activation='relu',padding='same')(x)
    x_shortcut = x
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x,x_shortcut])
    x = keras.layers.Conv2D(500, (3,3),activation='relu',padding='same')(x)
    x_shortcut = x
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x,x_shortcut])
    x = keras.layers.BatchNormalization()(x)
    out_flat= keras.layers.Flatten()(x)
    dense_1 = keras.layers.Dense(64 , activation='relu')(out_flat)
    out_1 = keras.layers.Dense(len(all_symbols) , activation='relu')(dense_1)
    model_out = keras.Model(inputs=inputs , outputs=out_1)
    model_out.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model_out

model1=model()
model1.fit(X,y,validation_split=0.33,epochs=50)
model.save('m1.h5')

model2=model()
model2.fit(X,y,validation_split==0.33,epoch=50)
model.save('m2.h5')
