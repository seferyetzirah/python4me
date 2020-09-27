# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:48:26 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""
#preprocessing:
    #scaling
import mglearn as mglearn
mglearn.plots.plot_scaling()
print("Different Scaling")
print("\n")
# Generate Dataset
#Preprocessing  Breast Cancer Data: MinMaxScaler
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print(X_train.shape)
print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
# transform data
X_train_scaled = scaler.transform(X_train)
print("MinMax Scaler Transformation")
# print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape)) 
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))
print("\n")
print("Next Transformation")
#transformation SVM
#APPLYING DATA TRANSFORMATIONS to test data
X_test_scaled = scaler.transform(X_test)
# print dataset properties before and after scaling
print("transformed shape: {}".format(X_test_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_test.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_test.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_test_scaled.max(axis=0)))

