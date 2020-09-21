# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:59:32 2020

@author: sefir
"""

import mglearn as mglearn
mglearn.plots.plot_animal_tree()
print("\n")

import sklearn.datasets as datasets
import pandas as pd
iris=datasets.load_iris()
#Creating the dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
from IPython.display import Image
from sklearn.tree import export_graphviz
export_graphviz(dtree, out_file="tree_iris.dot",
 filled=True, rounded=True,
 special_characters=True)
import graphviz
with open("tree_iris.dot") as f:
 dot_graph = f.read()
display(graphviz.Source(dot_graph))