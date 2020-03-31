import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import gc

dat=pd.read_csv('C:/Users/somna/Desktop/mosaic/DATA/driving_log.csv',names=list(range(7)),index_col=None)##your directory
imgadd=dat[[0,1,2]]
add=np.array(imgadd)
print(add)
ctrl=dat[[3,4,5,6]]
X=[]
for i in range(10):
    
    for j in range(3):
        tem=cv2.imread(str(add[i,j]).replace(" ",""))## to remove any error
        print(tem.shape)
        #tem=cv2.cvtColor(tem,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("frame"+str(i)+str(j),tem)
        tem=cv2.resize(tem,(64,32))
        #cv2.imshow("frame"+str(i)+str(j),tem)
        tem=cv2.GaussianBlur(tem,(3,3),0)

        if j==0:
            temp=tem
        elif j==1:
            temp=np.append(tem,temp,axis=1)
        else:
            temp=np.append(temp,tem,axis=1)
        
        #cv2.imshow("frame"+str(i)+str(j),tem)
        del tem
        gc.collect()
    cv2.imshow("frame"+str(i)+str(j),temp)
    cv2.waitKey(0)
    temp=temp.reshape(1,temp.shape[0],temp.shape[1],temp.shape[2])
    X.append(temp)
Xnp=np.array(X,dtype=np.float32)
yctrl=np.array(ctrl,dtype=np.float32)
yt=yctrl[:7,:]
ytt=yctrl[7:10,:]

print(yt.shape)
print(Xnp.shape)
    #temp=cv2.imread(str(imgadd[[0]].iloc[i]),0)
    #cv2.imshow("frame",temp)
    #cv2.waitKey(1)

print(Xnp.shape)
model=keras.Sequential([
    keras.layers.TimeDistributed(keras.layers.Conv2D(50,input_shape=(1,32,192,3),kernel_size=2,activation='tanh')),  ## using Conv
    keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid')),
    keras.layers.TimeDistributed(keras.layers.Flatten()),
    keras.layers.LSTM(200,dropout=0.2, recurrent_dropout=0.2,return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(15,activation='tanh')),
    keras.layers.TimeDistributed(keras.layers.Dense(4,activation='linear'))
    ])
model.compile(loss='mse', optimizer='adam',metrics = ['mse','mae'])

model.fit(Xnp[:7,:,:,:,:],yt,epochs=2)
print("model trained")
model.evaluate(Xnp[7:,:,:,:],ytt)

model.save("model001.h5")
#,input_shape=(32,192)
