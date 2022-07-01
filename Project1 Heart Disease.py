# -*- coding: utf-8 -*-
"""
@author: Alpha
"""

#loading dataset
import tensorflow as tf
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv(r'C:\Users\Aaron\anaconda3\envs\ai06.deeplearning\Projects\Project 1\archive\heart.csv')
data.head()

data.info()

data.isnull().any()

# **Model Preparation**

X = data.iloc[:,:13].values
y = data["target"].values
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3 , random_state = 0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 13, 
                     units = 8, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "relu", units = 14, 
                     kernel_initializer = "uniform"))
classifier.add(Dense(activation = "sigmoid", units = 1, 
                     kernel_initializer = "uniform"))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', 
                   metrics = ['accuracy'] )

classifier.fit(X_train , y_train , batch_size = 8 ,epochs = 100  )

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
cm

accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])
print(accuracy*100)

ann_acc_score = accuracy_score(y_test,y_pred)
print("Accuracy of ANN:",ann_acc_score*100,'\n')
print(classification_report(y_test,y_pred))
