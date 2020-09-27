# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:20:25 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""
#scaling test and train data the same way
import mglearn as mglearn
# Generate Dataset
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
from sklearn.model_selection import train_test_split
# split into test and training sets
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
#plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label='Test set', s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")
from sklearn.preprocessing import MinMaxScaler
#scale the data using MinMaxScalar
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                c=mglearn.cm2(0), label="Trainingset", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=mglearn.cm2(1), label="Test set",
s=60)
axes[1].set_title("Scaled Data")
#rescale the test set separately
# so test set min is 0 and test set max is 1
# DO NOT DO THIS! For illustration purposes only.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], c=mglearn.cm2(1),
label="Test set", s=60)
axes[2].set_title("Improperly Scaled Data")
for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    fig.tight_layout()