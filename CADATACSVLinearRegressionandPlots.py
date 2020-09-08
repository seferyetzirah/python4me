# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:20:58 2020

@author: sefir
"""


import numpy as np
from scipy import stats
import pandas as pd
cadataset= pd.read_csv("cadataexcelCSV.csv")
cadataset.describe()
cadataset.hist('total rooms')
import matplotlib.pyplot as plt
X=cadataset[['total rooms']].values
y=cadataset[['median house value']].values
plt.scatter(X,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33,
random_state=0)
from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
#results in a linear fit,  test set for prediction
y_pred=regressor.predict(X_test)
#create the linear fit using the train set
print("Linear Fit using Train set \n")
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel('total rooms')
plt.ylabel('median_house_value')
plt.show()
print("\n")
#apply linear fit to the test set
print("Linear fit for test set \n")
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Total Rooms vs. Median Hpuse Value (Test set)')
plt.xlabel('total rooms')
plt.ylabel('median_house_value')
plt.show()
