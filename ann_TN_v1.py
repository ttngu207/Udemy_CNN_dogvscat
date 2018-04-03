# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 13:01:06 2017

@author: ttngu207

General Architecture:
    
Input -> hidden_1 + activate -> ... -> hidden_n + activate -> Output


number of hidden layer
number of node per layer
type of activation function: sigmoid, ReLu, softmax
type of cost function: cross-entropy, MSE
regularization for overfitting: dropout (specify drop-out rate)
type of optimizer
epochs (interation)
batch size

k-fold cross-validation
grid-search for parameter tuning

Udemy ANN - Bank drop-out rate business problem
"""

# Installing Theano (pip install theano)
#   Numerical computation (based on numpy) - run on CPU + GPU
# Installing Tensorflow (pip install tensorflow)
#   Numerical computation (based on numpy) - run on CPU + GPU
# Installing Keras (pip install keras)
#   Deep learning Model - based on Theano + Tensorflow

# Part 1 - data processing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

''' ========================== Importing the dataset  ======================= ''' 
dataset = pd.read_csv('Churn_Modelling.csv')                              #panda DataFrame
X = dataset.iloc[:, 3:13].values #(for range: last element is exclusive)  #numpy ndarray
y = dataset.iloc[:, 13].values                                            # int64

''' ========================== Preprocess data ======================= '''        
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
country_encoder = LabelEncoder() # Encode "Country"
X[:, 1] = country_encoder.fit_transform(X[:, 1])
gender_encoder = LabelEncoder() # Encode "Gender"
X[:, 2] = gender_encoder.fit_transform(X[:, 2])
# convert [0,1,2] into [100],[010],[001] to denote not "ordinal" (e.g. 2 is not larger than 1) 
onehotencoder_1 = OneHotEncoder(categorical_features = [1])
X = onehotencoder_1.fit_transform(X).toarray()
# Remove 1 dummy variable (out of 3) to avoid the "dummy variable trap??"
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - per feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

''' ========================== Configure the neural-network model ======================= '''
# Create your classifier here
from keras.models import Sequential  # NN model type: sequential
from keras.layers import Dense       # NN layer type: densely-connected NN layer

# Initialize the NN model
classifier = Sequential() 

# Add hidden layer - acti fn is rectified linear 'relu'
classifier.add(Dense(input_dim=11 , units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))

classifier.add(Dense(units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))

# Add output layer - acti fn is sigmoid "sigmoid" (if multiple output, use "softmax")
classifier.add(Dense(units = 1, activation='sigmoid', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))

# Compile the ANN classifier.compile
# optimizer: learning method (e.g. stochastic gradient decent ('adam'), etc.)
# loss: loss-function, objective-function
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

''' ======================== Fit ANN to the training set ================ '''
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100) # epoch is like iteration number

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# binarize the prediction
thres = 0.5;
y_pred = (y_pred>thres);

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

''' HW - Apply to new data '''

newData2test = np.array([[600,'France','Male',40,3,60000,2,1,1,50000],
                         [500,'France','Male',20,3,15000,2,1,1,60000] ])

# preprocess
newData2test[:,1] = country_encoder.transform(newData2test[:,1])
newData2test[:,2] = gender_encoder.transform(newData2test[:,2])
newData2test = onehotencoder_1.transform(newData2test).toarray()
newData2test = newData2test[:,1:]
# scale
newData2test = sc.transform(newData2test)

# predict
new_pred = classifier.predict(newData2test)

''' ======================== K-fold Cross validation ================ '''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential  # NN model type: sequential
from keras.layers import Dense       # NN layer type: densely-connected NN layer
from keras.layers import Dropout

dropout_rate = 0.3
def Build_classifier():
    classifier = Sequential() 
    
    classifier.add(Dense(input_dim=11 , units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dropout(dropout_rate))
    
    classifier.add(Dense(units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dropout(dropout_rate))
    
    classifier.add(Dense(units = 1, activation='sigmoid', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Build_classifier, batch_size = 10, epochs = 100)
classifier.fit(X_train, y_train, batch_size = 100, epochs = 100) # epoch is like iteration number

if __name__ == "__main__":
    cv_acc = cross_val_score(classifier, X_train, y_train, cv = 10)   # cv: number of fold
    cv_acc = cross_val_score(classifier, X_train, y_train, cv = 10,n_jobs=-1)    
    



''' ======================== Parameter tuning ================ '''
# Grid search
from sklearn.model_selection import GridSearchCV

def Build_classifier(optimizer_type, dropout_rate):
    classifier = Sequential() 
    
    classifier.add(Dense(input_dim=11 , units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dropout(dropout_rate))
    
    classifier.add(Dense(units = 6, activation='relu', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.add(Dropout(dropout_rate))
    
    classifier.add(Dense(units = 1, activation='sigmoid', use_bias=True, bias_initializer='zeros', kernel_initializer = 'glorot_uniform'))
    classifier.compile(optimizer=optimizer_type,loss='binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = Build_classifier)


hyper_parameters = {'batch_size':[16,32],
                    'epochs':[100,200],
                    'dropout_rate':[0.1,0.2,0.3],
                    'optimizer_type':['adam','rmsprop']}

if __name__ == "__main__":
    grid_search = GridSearchCV(classifier,hyper_parameters,scoring='accuracy',cv=10)
    grid_result = grid_search.fit(X_train,y_train)

















