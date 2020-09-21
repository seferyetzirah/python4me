# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:08:06 2020

@author: sefir
"""
#unpruned breast cancer set tree

import sklearn.datasets as datasets
import pandas as pd
cancer=datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
stratify=cancer.target, random_state=42)
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#pre pruning the data

import sklearn.datasets as datasets
import pandas as pd
cancer=datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
stratify=cancer.target, random_state=42)
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(max_depth= 4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:,.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#analyze visualize the tree

from sklearn.tree import export_graphviz
# We can visualize the tree using the export_graphviz function from the tree module
export_graphviz(tree, out_file="tree.dot", 
                impurity=False, filled=True)
#This writes the data in the .dot file - a text file for storing graphs.
#Set an option to color the nodes to majority class in each node and pass the class and feature names
import graphviz
with open("tree.dot") as f:
 dot_graph = f.read()
display(graphviz.Source(dot_graph))

#Feature Importance method

print("Feature importance: \n{}".format(tree.feature_importances_))
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
 n_features = cancer.data.shape[1]
 plt.barh(range(n_features), model.feature_importances_, align='center')
 plt.yticks(np.arange(n_features), cancer.feature_names)
 plt.xlabel("Feature importance")
 plt.ylabel("Fetaure")
 plt.ylim(-1, n_features)

plot_feature_importances_cancer(tree)

# another look, at features
import mglearn as mglearn
tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

