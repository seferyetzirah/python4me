# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:23:28 2020

@author: Ian Saltzman
is721863@sju.edu
Diabetes Database from Pima Indians from Kaggle
Linear SVC

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs

df= pd.read_csv('diabetes.csv')
print(df)
print("Diabetes.keys():\n", df.keys())
print("\n")
print("Shape of Data:", df.shape)
print("\n")

print("Decision boundaries with a 3 v 1 or 1 vs the rest data set classifier")

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome', axis=1))
scaled_data = pd.DataFrame(df, columns = df.drop('Outcome', axis=1).columns)

X = scaled_data
y = df['Outcome']

from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

X, y = make_blobs()
linear_svm = LinearSVC().fit(X, y)


print("Coefficient shape: ", linear_svm.coef_.shape)
print("Intercept shape: ", linear_svm.intercept_.shape)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
 plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("All other variables")
plt.ylabel("Outcome")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
