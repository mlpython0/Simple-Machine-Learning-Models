# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:58:00 2020

@author: Jay Shah
"""

#Importing libraries
import numpy as np
import pandas as pd

#Importing the dataset
columns = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'] 
dataset_orig = pd.read_csv(r'C:\Users\Jay Shah\Downloads\abalone.data', header = None, names = columns )

#Handling the categorical variables
dataset_orig = pd.get_dummies( dataset_orig, drop_first = True)

#Splitting the Dependent and Independent Variables
X = dataset_orig.iloc[:,[0,1,2,3,4,5,6,8,9]].values
y = dataset_orig.iloc[:,7].values

#Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Using PCA for Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
#explained_ratio = pca.explained_variance_ratio_



#Building the model
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 0)
regressor_rf.fit(X_train_pca, y_train)

#Predicting the results
y_pred = regressor_rf.predict(X_test_pca)

#Analyzing the results
accuracy = regressor_rf.score(X_test_pca, y_test)#Accuracy is 56.97%

#Optimizing the model 
#Using Backward Elimination To build Optimum Model
import statsmodels.regression.linear_model as lm
X = np.append(arr = np.ones((4177,1)).astype(float), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 8]]
regressor_opt = lm.OLS(endog = y, exog = X_opt).fit()
regressor_opt.summary()


X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]] #Eliminated X[9]
regressor_opt = lm.OLS(endog = y, exog = X_opt).fit()
regressor_opt.summary()

#We get the optimum model
#We check the accuracy for the optimum model
from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)
regressor_rf_opt = RandomForestRegressor(n_estimators = 1000, criterion = 'mae', random_state = 0)
regressor_rf_opt.fit(X_train_opt, y_train_opt)

#Predicting the results
y_pred_opt = regressor_rf_opt.predict(X_test_opt)

#Analyzing the results
accuracy_opt = regressor_rf_opt.score(X_test_opt , y_test_opt)#Accuracy is 57.39%


# #The Linear Regression Model
# #The accuracy achieved with model is 53.90%
# #Building the Model
# from sklearn.linear_model import LinearRegression
# regressor_lr = LinearRegression()
# regressor_lr.fit(X_train_opt, y_train_opt)

# #Predicting the result
# y_pred_lr = regressor_lr.predict(X_test_opt)

# #Analyzing the model
# accuracy_lr = regressor_lr.score(X_test_opt, y_test_opt)


# #The SVR Model
# #The accuracy achieved with model is 52.50%
# from sklearn.model_selection import train_test_split
# X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]] 
# X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)


# from sklearn.svm import SVR
# regressor_svr = SVR()
# regressor_svr.fit(X_train_opt, y_train_opt)

# #Predicting the result
# y_pred_svr = regressor_svr.predict(X_test_opt)

# #Analyzing the model
# accuracy_svr = regressor_svr.score(X_test_opt , y_test_opt)

"""
We do not get satisfying results from the Linear Regression Model and SVR Model.
Therefore, it turns out that Random Forest Regression gives the most
powerful performance on our Optimum dataset with an accuracy of 57.39%
"""




