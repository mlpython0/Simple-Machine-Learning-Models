# -*- coding: utf-8 -*-
"""


@author: Jay Shah
"""

import numpy as np
import pandas as pd


#Importing the dataset
dataset = pd.read_csv(r'C:\Users\Jay Shah\Desktop\Machine Learning\Balance Scale\balance-scale.data', header = None)


#DIfferentiating the dependent and independent features
X = dataset.iloc[:,1:].values
y_ = dataset.iloc[:,0]

#Handling the categorical variables in dependent dataset
y = pd.get_dummies(y_, drop_first = True)
y = y.iloc[:,:].values

"""
In the dependent dataset, 1 corresponds to R(Right)
and 0 corresponds to L(Left)
"""


#Splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 4000)
classifier_rf.fit(X_train, y_train)
y_pred = classifier_rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

