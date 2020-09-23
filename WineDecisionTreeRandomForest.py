# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 19:33:34 2020

@author: @seferyetzirah is721863@sju.edu
"""

import sklearn.datasets as datasets
import pandas as pd


from sklearn.datasets import load_wine
wine = load_wine()
print(wine.DESCR)
print("\n")
print(wine.data.shape)
print("\n")
print(wine.feature_names)
print("\n")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target,
    stratify=wine.target)
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
print("\n")

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file="tree.dot",  class_names=["class 0", "class 1", "class2"], 
            feature_names= wine.feature_names, impurity=False, filled=True)
#This writes the data in the .dot file - a text file for storing graphs.
#Set an option to color the nodes to majority class in each node and pass the class and feature names
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

print("Random Forest for Wine")
print("\n")
from sklearn.ensemble import RandomForestClassifier
import mglearn
import matplotlib.pyplot as plt


forest=RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print("\n")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
import numpy as np

importances = forest.feature_importances_
features = wine['feature_names']
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
