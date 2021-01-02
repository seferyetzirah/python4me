# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 08:24:23 2021

@author: sefir
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



df= pd.read_csv('ptvn1.CSV')
print(df)
print("\n")
print("Summary Statistics\n")
print(df.describe())
print("\n")
print("Sum\n")
print(df.sum())
print("\n")
print("Median\n")
print(df.median())
print("\n")



X=df[['P.NH3']].values
y=df[['PH2.DM.TVN']].values



lr = LinearRegression().fit(X, y)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X, y)))
print("Test set score: {:.2f}".format(lr.score(X, y)))

corM = df.corr()
print("Correl Coeff: {}".format(df.corr()))

import seaborn as sns
df_small = df.iloc[:,:11]
correlation_mat = df_small.corr()
sns.heatmap(correlation_mat, annot= True)
plt.show()
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33,
#random_state=0)
#from sklearn import linear_model

#lr = LinearRegression().fit(X_train, y_train)
#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))




#regressor = linear_model.LinearRegression()
#regressor.fit(X_train, y_train)
#results in a linear fit,  test set for prediction
#y_pred=regressor.predict(X_test) 