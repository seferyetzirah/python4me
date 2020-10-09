# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:08:51 2020

@author: Ian Saltzman 
        is728163@sju.edu
        Machine Learning: Representing Data and Engineering Features
"""

#USING EVALUATION METRICS IN MODEL SELECTION
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn .metrics import roc_auc_score
from sklearn .metrics import roc_curve
from sklearn.model_selection import GridSearchCV
digits = load_digits()
#creating a 9:1 imbalanced dataset from the digits by classifying 9
#against the nine other classes
y = digits.target == 9
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 digits.data, y, random_state=0)
# We provide a somewhat bad grid to illustrate the point:
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}
# using the default scoring of accuracy:
grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)
print("Grid-Search with accuracy")
print("Best parameters:", grid.best_params_)
from sklearn.metrics import roc_auc_score
print("Best cross-validation score (accuracy): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
 roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
# Using AUC Scoring instead
grid = GridSearchCV(SVC(), param_grid=param_grid, scoring ="roc_auc")
grid.fit(X_train, y_train)
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
from sklearn.metrics import roc_auc_score
print("Best cross-validation score (accuracy): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
 roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
from sklearn.metrics.scorer import SCORERS
print("Availablescorers: \n{}".format(sorted(SCORERS.keys())))