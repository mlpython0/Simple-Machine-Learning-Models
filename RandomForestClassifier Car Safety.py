# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:29:52 2020

@author: Jay Shah
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


#Importing the dataset
dataset_orig = pd.read_csv(r'C:\Users\Jay Shah\Downloads\car.data', sep = ",", header = None)#For Reference
dataset = pd.read_csv(r'C:\Users\Jay Shah\Downloads\car.data', sep = ",", header = None)

#Handling the categorical variables
from sklearn.preprocessing import LabelEncoder
col = [0,1,2,3,4,5,6]
label_encoder = LabelEncoder()
for i in col:
    dataset.iloc[:,i] = label_encoder.fit_transform(dataset.iloc[:,i])
   
#Splitting the dataset into independent and dependent features
X = dataset.iloc[:,0:6]
y = dataset.iloc[:,-1]

#One hot Encoding the independent and dependent features
X_ = pd.get_dummies(X, drop_first = True)
y_ = pd.get_dummies(y, drop_first = True)

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size = 0.2, random_state = 0)

#Building the Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 100)
classifier_rf.fit(X_train, y_train)

#Making the prediction using the Random Forest Classifier
y_pred_rf = classifier_rf.predict(X_test)

#Analyzing accuracy score for the classifier_rf
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred_rf)

