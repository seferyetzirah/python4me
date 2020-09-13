# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:43:08 2020

@author: sefir
"""

#Lasso L1 Regularization
from sklearn.linear_model import Lasso
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy
from sklearn import linear_model
import mglearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

X, y = mglearn.datasets.load_extended_boston()
print("X.shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter"
# Otherwise the model would warn us that we should increase max_iter
#alpha 0.01
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

#lasso with alpha 0.0001

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
      
#look at multiple coefficient magnitudes


