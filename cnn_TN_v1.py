# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:12:45 2017

@author: ttngu207

General Architecture:
    
Input -> Convolv_1 + activate -> pool_1 -> Convolv_2 + activate -> pool_2 -> ... -> Convolv_n + activate -> pool_n
      -> Flattening -> hidden_1 + activate -> ... -> hidden_n + activate -> Output

"""

topdir = 'C:\\Users\\ttngu207\\OneDrive\\Python Learning\\Deep Learning A-Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Section 4 - Building an ANN\\'
dogcat_traindatadir = 'D:\\Python Scripts\\Cat Dog dataset CNN\\dataset\\training_set\\'
dogcat_testdatadir = 'D:\\Python Scripts\\Cat Dog dataset CNN\\dataset\\test_set\\'

'=========================== Import Packages ========================= '
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import keras.layers
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, Dropout

'=========================== Data preprocessing, conditioning ====================='

'---- Get the train set, the manual way... ----'
import imageio
img = imageio.imread(dogcat_traindatadir+'dogs\\'+'dog.288.jpg')

'=========================== Configure the ConvNet ====================='

def makeConvNet(input_shape, conv_filters, nn_units, kernel_size=(3,3),\
                regulizationParam = 0, dropoutRate=0,\
                optimizer='adam',loss='binary_crossentropy'):
    from keras import regularizers
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, Activation, Dropout, MaxPool2D, Flatten
    
    M = Sequential()
    '---- Add the Convolution layers + Dropout + Pooling ---- '
    for index, k in enumerate(conv_filters):
        if index == 0:
            M.add(Conv2D(filters=k,kernel_size=kernel_size,input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l1(regulizationParam) ))
        else:
            M.add(Conv2D(filters=k,kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l1(regulizationParam)))
        M.add(MaxPool2D(pool_size=(2,2)))
    '---- After all the conv layers, flatten and add a final fully-connected NN layer ----'    
    M.add(Flatten())

    for j in nn_units[0:-1]:
        M.add(Dense(units=j,activation='relu', kernel_regularizer=regularizers.l1(regulizationParam)))
        M.add(Dropout(rate=dropoutRate))
        
    '---- Add a final output layer ----'    
    M.add(Dense(units=nn_units[-1],activation='sigmoid'))
    
    '---- Compile ---- '    
    M.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

    return M

'---- Create the ConvNet model ----'
input_image_size =(150,150,3)

ConvNet_arg = {'input_shape':input_image_size, 'conv_filters':[32,32,64,64], 'nn_units':[64,64,64,1],\
               'kernel_size':(3,3), 'regulizationParam':0.00, 'dropoutRate':0.3, \
               'optimizer':'adam', 'loss':'binary_crossentropy'}

model = makeConvNet(**ConvNet_arg)

'=========================== Build the ImageDataGenerator ====================='
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32;

ImgGen_trainData = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,\
                            rotation_range=0.2, width_shift_range=0.1, height_shift_range=0.1,\
                            shear_range=0.2, zoom_range=0.2, rescale = 1/255, horizontal_flip=True)
ImgGen_trainData = ImgGen_trainData.flow_from_directory(dogcat_traindatadir,\
                                     target_size=input_image_size[0:2],\
                                     class_mode='binary',\
                                     batch_size=batch_size)
ImgGen_testData = ImageDataGenerator(rescale = 1/255)
ImgGen_testData = ImgGen_testData.flow_from_directory(dogcat_testdatadir,\
                                     target_size=input_image_size[0:2],\
                                     class_mode='binary',\
                                     batch_size=batch_size)

'=========================== Build the ImageDataGenerator ====================='
history = model.fit_generator(generator=ImgGen_trainData,\
                    steps_per_epoch = ImgGen_trainData.samples/ImgGen_trainData.batch_size,\
                    epochs=1,\
                    validation_data = ImgGen_testData, \
                    validation_steps = ImgGen_testData.samples/ImgGen_testData.batch_size, \
                    workers=6)
   
'''
'---- Predict on an image ----'
from PIL import Image as pil_image
dogcat_predictdatadir = 'D:\\Python Scripts\\Cat Dog dataset CNN\\dataset\\single_prediction\\'
img1 = pil_image.open(dogcat_predictdatadir+'cat_or_dog_1.jpg')
img2= pil_image.open(dogcat_predictdatadir+'cat_or_dog_2.jpg')

img1 = np.array(img1.resize(input_image_size[0:2])) * (1/255)
img2 = np.array(img2.resize(input_image_size[0:2])) * (1/255)


model.predict(np.array([img1,img2]))
model.predict_classes(np.array([img1,img2]))
'''
'''
'=========================== Tunning the CNN ====================='

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

ConvM = KerasClassifier(build_fn=makeConvNet,**ConvNet_arg)
ConvGrid = GridSearchCV(estimator=ConvM, param_grid={})
'''
            










            
            












