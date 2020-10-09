# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 15:35:42 2020

@author: Ian Saltzman
    is721863@sju.edu
"""

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn as mglearn


wine = load_wine()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


X_trainval, X_test, y_trainval, y_test = train_test_split(wine.data, wine.target)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval,
 random_state=1)
print("Size of training set: {} size of validation set: {} size of test set:"
 "{}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        
        svm = SVC(gamma=gamma, C=C)
        
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
       
        score = np.mean(scores)
       
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
 random_state=0)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score {:.2f}".format(test_score))
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
 random_state=0)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score {:.2f}".format(test_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))