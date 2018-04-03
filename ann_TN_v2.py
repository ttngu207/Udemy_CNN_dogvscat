# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:33:10 2018

Redo fully-connected neural network

@author: ttngu207
"""

' ============== Import packages ========'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

' ============== Load data =============='
topdir = 'C:\\Users\\ttngu207\\OneDrive\\Python Learning\\Deep Learning A-Z\\Volume 1 - Supervised Deep Learning\\Part 1 - Artificial Neural Networks (ANN)\\Section 4 - Building an ANN\\'
data_filename = 'Churn_Modelling.csv'

dset = pd.read_csv(topdir+data_filename)

# Touch up the data set a bit
dset.Geography = dset.Geography.astype('category')
dset.Gender = dset.Gender.astype('category')

' ========================== Feature extraction ======================= '
Feature_Label = dset[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']]

# Encode "categorical" data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Geo_Cate_Encoder = LabelEncoder()
Feature_Label.Geography = Geo_Cate_Encoder.fit_transform(Feature_Label.Geography)

Gender_Cate_Encoder = LabelEncoder()
Feature_Label.Gender = Gender_Cate_Encoder.fit_transform(Feature_Label.Gender)

X = np.array(Feature_Label.iloc[:,0:-1])
y = np.array(Feature_Label.iloc[:,-1])
# non-ordinal
OneHot = OneHotEncoder(categorical_features=[1]) 
X = OneHot.fit_transform(X).toarray()
# remove 1 dummy variable from Geography to avoid the dummy variable trap
X = X[:,1:]

' ---- Split Train Test ----'
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2, random_state=1)

' ---- Feature scaling ----'
from sklearn.preprocessing import StandardScaler
FeatureScaler = StandardScaler()
Xtrain = Xscaler.fit_transform(Xtrain)
Xtest = Xscaler.transform(Xtest)

' ========================== Configure the neural-network model ======================= '
# Create your classifier here
from keras.models import Sequential  # NN model type: sequential
from keras.layers import Dense       # NN layer type: densely-connected NN layer
from keras.layers import Activation, Dropout

dropout_rate = 0;
' Configure the classifer '
Clsfier = Sequential()

Clsfier.add(Dense(input_dim=11,units=20))
Clsfier.add(Activation('relu'))
Clsfier.add(Dropout(dropout_rate))

Clsfier.add(Dense(units=10))
Clsfier.add(Activation('relu'))
Clsfier.add(Dropout(dropout_rate))

Clsfier.add(Dense(units=4))
Clsfier.add(Activation('relu'))
Clsfier.add(Dropout(dropout_rate))

Clsfier.add(Dense(units=1))       
Clsfier.add(Activation('sigmoid'))

' Compile the classifier '

Clsfier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])

Clsfier.fit(Xtrain,ytrain,batch_size=100,epochs=100)
Clsfier.evaluate(Xtest,ytest)

' Implement K-fold cross validation '
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def Cmodel(input_dim,layerUnits,dropout_rate=0):
    from keras.models import Sequential  # NN model type: sequential
    from keras.layers import Dense, Activation, Dropout      # NN layer type: densely-connected NN layer
    
    Clsfier = Sequential()
    for indx,k in enumerate(layerUnits[0:-1]):
        if indx == 0:
            Clsfier.add(Dense(input_dim=input_dim,units=k))
        else:
            Clsfier.add(Dense(units=k))
        Clsfier.add(Activation('relu'))
        Clsfier.add(Dropout(dropout_rate))
    
    Clsfier.add(Dense(units=layerUnits[-1]))       
    Clsfier.add(Activation('sigmoid'))
    
    Clsfier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return Clsfier


model = KerasClassifier(build_fn=Cmodel,input_dim=11,layerUnits=[20,10,4,1],dropout_rate=dropout_rate, \
                        batch_size=100, epochs = 100)

score = cross_val_score(model,X=Xtrain,y=ytrain,cv=10)

testpredict = model.predict(Xtest)

' Tuning using GridSearch '
from sklearn.model_selection import GridSearchCV

model = KerasClassifier(build_fn=Cmodel,input_dim=11, \
                        batch_size=200, epochs=100)

layerUnit_grid = [[20,10,5,1],\
                  [8,8,1],\
                  [8,4,1],\
                  [20,1]]

GridS = GridSearchCV(estimator=model,param_grid = {'layerUnits':layerUnit_grid,'dropout_rate':[0,0.25]},cv=5)
GridS.fit(Xtrain,ytrain)


























