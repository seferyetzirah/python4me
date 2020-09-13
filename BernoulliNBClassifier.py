# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:30:49 2020

@author: sefir
"""
#Bernoulli Naive Bayes Classifiers

import numpy as np
X = np.array([[0, 1, 0, 1],
 [1, 0, 1, 1],
 [0, 0, 0, 1],
 [1, 0, 1, 0]])
y= np.array([0, 1, 0, 1])
15
# 4 data points with 4 binary features each
# Two classes 0 and 1
counts = {}
for label in np.unique(y):
 # iterate over each class
 #count (sum) entries of 1 per feature
 counts[label] = X[y == label].sum(axis=0)
print("Feature counts: \n{}".format(counts))

#Titanic Dataset NB Models

