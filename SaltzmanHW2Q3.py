# -*- coding: utf-8 -*-
"""
Standard format for MlEARNING  Sep 05 2020 21:06:45
@author: Ian Saltzman is721863@sju.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split

X, y =np.arange(40).reshape((10,4)),range(1,)
print("X:\n{}".format(X))
print("y:\n{}".format(list(y)))
print("\n")
print("Test Train Data: 0.2 and 0.8\n")
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2,
random_state=0, shuffle=False)
print("X_train:\n{}".format(X_train))
print("y_train:\n{}".format(y_train))
print("X_test:\n{}".format(X_test))
print("y_test:\n{}".format(y_test))
from sklearn import linear_model
regressor= linear_model.LinearRegression()
print("Linear fit\n")
regressor.fit(X_train, y_train)
print("Test set prediction\n")
y_pred= regressor.predict(X_test)
