# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:10:07 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
DBSCAN
"""

#DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS AND NOISE (DBSCAN)
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))

#Cluster assignments for different values of min_samples and epochs
mglearn.plots.plot_dbscan()

#DBSCAN on Two Moons Data

import mglearn as mglearn
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


