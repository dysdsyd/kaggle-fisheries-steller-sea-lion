#Solve sea lions counting problem as regression problem on whole image

#the initial model adapted to blending


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Input, concatenate, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import cv2
import sys
import os
import matplotlib.pyplot as plt

n_classes= 5
batch_size= 1
epochs= 100
image_size= 1024
model_name= 'covnet_v2_dropout_batchnorm1024batch1'#blending vertion

dir_path = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/train_images_1024x1024'
dir_pathTest = 'F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/test_images_1024x1024'
weightsPath = "C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/experiment-charles/layer1/weights/" + model_name + "_" + str(epochs) + '_model.h5'
outputPathTest =  "C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/experiment-charles/layer1/outputs/" + model_name + "_" +  str(epochs) + '_Testsubmission.csv'
outputPathBlend =  "C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/experiment-charles/layer1/outputs/" + model_name + "_" +  str(epochs) + '_Blendsubmission.csv'


def read_training_list():
    training_list = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/IDsTrainingSet.txt')
    return training_list
#print(read_training_list())


def read_blending_list():
    blending_list = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/IDsforBlending.txt')
    return blending_list

#Just remove images from mismatched_train_images.txt
def load_data(dir_path):
    df = pd.read_csv('C:/Users/Charles/OneDrive/DS/Kaggle/NOAA Fisheries Steller Sea Lion Population Count/counts/train.csv')
    training_list= read_training_list()
        
    image_list=[]
    y_list=[]
    for i in training_list:
        image_path= os.path.join(dir_path, str(i)+'.png')
        print(image_path)
        img= cv2.imread(image_path)
        print('i4mg.shape',img.shape)
        image_list.append(img)
        
        row= df.ix[int(i)] 
        y_row= np.zeros((5))
        y_row[0]= row['adult_males']
        y_row[1]= row['subadult_males']
        y_row[2]= row['adult_females']
        y_row[3]= row['juveniles']
        y_row[4]= row['pups']
        y_list.append(y_row)
            
    x_train= np.asarray(image_list)
    y_train= np.asarray(y_list)
    
    print('x_train.shape', x_train.shape)
    print('y_train.shape', y_train.shape)

    return x_train,y_train
    
#load_data('F:/DS-main/Kaggle-main/NOAA Fisheries Steller Sea Lion Population Count - inputs/train_images_512x512')

def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_model():
    input_shape = (image_size, image_size, 3)
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
        
    model.add(Conv2D(n_classes, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling2D())
    
    print (model.summary())
    #sys.exit(0) #

    model.compile(loss=keras.losses.mean_squared_error,
            optimizer= keras.optimizers.Adadelta())
             
    return model

def train():
    model= get_model()
    
    x_train,y_train= load_data(dir_path)
    
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)
        
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

   
    model.save(weightsPath)
 

   
    
    
def re_train():
    model = load_model(weightsPath)
    
    x_train,y_train= load_data(dir_path)
    
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True)
        
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(x_train) / batch_size, epochs=200)

   
    model.save(weightsPath)
 

        
    
    

def create_submission():
    model = load_model(weightsPath)
    
    n_test_images= 18636
    pred_arr= np.zeros((n_test_images,n_classes),np.int32)
    for k in range(0,n_test_images):
        image_path= dir_pathTest + '/' + str(k) + '.png'
        print(image_path)
        img= cv2.imread(image_path)
        img= img[None,...]
        pred= model.predict(img, batch_size = 16, verbose=1)
        pred= pred.astype(int)
        pred_arr[k,:]= pred

    print('pred_arr.shape', pred_arr.shape)
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id']= range(0,n_test_images)
    df_submission['adult_males']= pred_arr[:,0]
    df_submission['subadult_males']= pred_arr[:,1]
    df_submission['adult_females']= pred_arr[:,2]
    df_submission['juveniles']= pred_arr[:,3]
    df_submission['pups']= pred_arr[:,4]
    df_submission.to_csv(outputPathTest,index=False)
    
    
def create_submissionBlend():
    model = load_model(weightsPath)
        
    blending_list= read_blending_list()
    pred_arr= np.zeros((len(list(blending_list)),n_classes),np.int32)
    for num, i in enumerate(blending_list):
        #print(type(i))
        image_path= os.path.join(dir_path, str(i)+'.png')
        print(image_path)
        img= cv2.imread(image_path)
        img= img[None,...]
        pred= model.predict(img, batch_size = 16, verbose=1)
        pred= pred.astype(int)
        pred_arr[num,:]= pred
    print('pred_arr.shape', pred_arr.shape)
    pred_arr = pred_arr.clip(min=0)
    df_submission = pd.DataFrame()
    df_submission['test_id']= list(blending_list)
    df_submission['adult_males']= pred_arr[:,0]
    df_submission['subadult_males']= pred_arr[:,1]
    df_submission['adult_females']= pred_arr[:,2]
    df_submission['juveniles']= pred_arr[:,3]
    df_submission['pups']= pred_arr[:,4]
    df_submission.to_csv(outputPathBlend,index=False)

train()

#re_train()
        
create_submission()
create_submissionBlend()



















