import pandas as pd
import cv2
import numpy as np
from sklearn.utils import shuffle
import os
from collections import deque
import copy
import matplotlib
import matplotlib.pyplot as plt
# from keras.utils import np_utils

# from config import Config

class ActionDataGenerator(object):
    
    def __init__(self,root_data_path,temporal_stride=1,temporal_length=32,resize=112):
        
        self.root_data_path = root_data_path
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.resize=resize
    def file_generator(self,data_path,data_files):
        '''
        data_files - list of csv files to be read.
        '''
        for f in data_files:       
            tmp_df = pd.read_csv(os.path.join(data_path,f))
            label_list = list(tmp_df['Label'])
            total_images = len(label_list) 
            if total_images>=self.temporal_length:
                num_samples = int((total_images-self.temporal_length)/self.temporal_stride)+1
#                 print ('num of samples from vid seq-{}: {}'.format(f,num_samples))
                img_list = list(tmp_df['FileName'])
            else:
#                 print ('num of frames is less than temporal length; hence discarding this file-{}'.format(f))
                continue
            
            start_frame = 0
            samples = deque()
            samp_count=0
            for img in img_list:
                samples.append(img)
                if len(samples)==self.temporal_length:
                    samples_c=copy.deepcopy(samples)
                    samp_count+=1
                    for t in range(self.temporal_stride):
                        samples.popleft() 
                    yield samples_c,label_list[0]

    def load_samples(self,data_cat='train'):
        data_path = os.path.join(self.root_data_path,data_cat)
        csv_data_files = os.listdir(data_path)
        file_gen = self.file_generator(data_path,csv_data_files)
        iterator = True
        data_list = []
        while iterator:
            try:
                x,y = next(file_gen)
                x=list(x)
                data_list.append([x,y])
            except Exception as e:
#                 print ('the exception: ',e)
                iterator = False
#                 print ('end of data generator')
        return data_list
    
    def shuffle_data(self,samples):
        data = shuffle(samples,random_state=2)
        return data
    
    def preprocess_image(self,img):
        img = cv2.resize(img,(self.resize,self.resize))
        img = img/255
        return img
    
    def data_generator(self,data,batch_size=10,shuffle=True):              
        """
        Yields the next training batch.
        data is an array [[img1_filename,img2_filename...,img16_filename],label1], [image2_filename,label2],...].
        """
        num_samples = len(data)
        if shuffle:
            data = self.shuffle_data(data)
        while True:   
            for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                batch_samples = data[offset:offset+batch_size]
                # print(len(batch_samples))
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for batch_sample in batch_samples:
                    # Load image (X)
                    x = batch_sample[0]
                    # print("yyyyyy",x)
                    y = batch_sample[1]
                    temp_data_list = []
                    for img in x:
                        
                        try:
                            img = cv2.imread(img)
                            #apply any kind of preprocessing here
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                            
                            # print(img.shape,"rrrrr")
                            
                            img = self.preprocess_image(img)
                            r = img.shape[0]
                            img = img.reshape(r,r,1)
                            temp_data_list.append(img)
    
                        except Exception as e:
                            print (e)
#                             print ('error reading file: ',img)  
    
                    # Read label (y)
                    #label = label_names[y]
                    # Add example to arrays
                    dummy_arr = np.array(temp_data_list)
                    # s = dummy_arr.shape[1]
                    dummy_array = dummy_arr.reshape(dummy_arr.shape[0],-1).T

                    u, s, vh = np.linalg.svd(dummy_array, full_matrices=False)
                    dummy_array = u[:,0:8]
                    dummy_array = dummy_array.T
                    dummy_array = dummy_array.reshape(8,112,112,1)
#                     print(dummy_array.shape)

                    X_train.append(dummy_array)
                    y_train.append(y)
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
                #X_train = np.rollaxis(X_train,1,4)
                y_train = np.array(y_train)
                y_train = np.eye(3)[y_train]

                # The generator-y part: yield the next training batch            
                yield X_train, y_train




import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K 

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Activation, Conv3D, Input,Dense, Dropout, Flatten,
                          MaxPooling3D)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam




# from tensorflow.keras.models import Sequential
def actmodel(num_classes=3):
    # Define model
    # model = Sequential()
    x_i = Input(shape=(8,112,112,1))
    x = Conv3D(32, kernel_size=(3, 3, 1),  padding='same')(x_i)
    x= Activation('relu')(x)
    x = Conv3D(32, kernel_size=(3, 3, 1), padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPooling3D(pool_size=(3, 3, 1), padding='same')(x)
    x=Dropout(0.25)(x)

    x=Conv3D(64, kernel_size=(3, 3, 1), padding='same')(x)
    x=Activation('relu')(x)
    x=Conv3D(64, kernel_size=(3, 3, 1), padding='same')(x)
    x=Activation('relu')(x)
    x=MaxPooling3D(pool_size=(3, 3, 1), padding='same')(x)
    x=Dropout(0.25)(x)

    x=Flatten()(x)
    x=Dense(512, activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(num_classes)(x)
    y=Activation("softmax")(x)

    model = Model(inputs = x_i,outputs=y)
    model.summary()
    return model


model = actmodel()


root_data_path = '/home/ubuntu/ucf_model/data_files/'

data_gen_obj=ActionDataGenerator(root_data_path,temporal_stride=4,temporal_length=32)


train_data = data_gen_obj.load_samples(data_cat='train')

test_data = data_gen_obj.load_samples(data_cat='test')


print('num of train_samples: {}'.format(len(train_data)))
print('num of test_samples: {}'.format(len(test_data)))


train_generator = data_gen_obj.data_generator(train_data,batch_size=2,shuffle=True)

test_generator = data_gen_obj.data_generator(test_data,batch_size=2,shuffle=True)


model.compile(loss=categorical_crossentropy,
              optimizer=Adam(), metrics=['accuracy'])


# Fit model using generator
hist = model.fit(train_generator, 
                steps_per_epoch=len(train_data),validation_data = test_generator,validation_steps=len(test_data),epochs=1)