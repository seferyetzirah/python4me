# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:49:05 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""
#Looking at something with three sources

import mglearn as mglearn
S = mglearn.datasets.make_signals()
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("Time")
plt.ylabel("Signal")

#Decomposition of the three sources
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
S = mglearn.datasets.make_signals()
# mix data into a 100-dimensional state
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print("Shape of measurements: {}".format(X.shape))
from sklearn.decomposition import NMF
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print("Recovered signal shape: {}".format(S_.shape))
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)
models = [X, S, S_, H]
names = ['Observations (first three measurements)', 'True Sources', 'NMF recovered signals',
'PCA recovered signals']
fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .4}, subplot_kw={'xticks': (),
'yticks': ()})
for model, name, ax in zip(models, names, axes):
 ax.set_title(name)
 ax.plot(model[:, :3], '-')
 
