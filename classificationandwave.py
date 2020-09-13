# -*- coding: utf-8 -*-
"""
Standard format for MlEARNING  Sep 11 22:08:01 2020
@author: Ian Saltzman is721863@sju.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split



#KNN FORGE
#Using mglearn library and one nearest neighbor
# Generate Dataset
import mglearn as mglearn
X, y = mglearn.datasets.make_forge()
mglearn.plots.plot_knn_classification(n_neighbors=1)

#Using mglearn library using three closest neighbors
#n is 3
# Generate Dataset
import mglearn as mglearn
X, y = mglearn.datasets.make_forge()
mglearn.plots.plot_knn_classification(n_neighbors=3)

#KNN WITH SCIKIT LEARN
# Generate Dataset
import mglearn as mglearn
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("test set predictions: {}".format(clf.predict(X_test)))
print("test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

#K Nearest Neighbors Classifier --- Two dimensional datasets for xyplane prediction
# Generate Dataset
import mglearn as mglearn
X, y = mglearn.datasets.make_forge()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
 # the fit method returns the object self, so we can instantiate
 # and fit in one line
 from sklearn.neighbors import KNeighborsClassifier
 clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("{} neighbor(s)".format(n_neighbors))
 ax.set_xlabel("feature 0")
 ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

#Model complexity example with the breast cancer data

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
from sklearn.neighbors import KNeighborsClassifier
for n_neighbors in neighbors_settings:
 # build the model
 clf = KNeighborsClassifier(n_neighbors=n_neighbors)
 clf.fit(X_train, y_train)
 # record training set accuracy
 training_accuracy.append(clf.score(X_train, y_train))
 # recording generalization accuracy
 test_accuracy.append(clf.score(X_test, y_test))
import matplotlib.pyplot as plt
plt.plot(neighbors_settings, training_accuracy, label = "training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()