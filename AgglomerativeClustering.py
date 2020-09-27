# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:04:23 2020
@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learnign
Agglomerative clustering
"""
#Agglomerative clustering

import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
mglearn.plots.plot_agglomerative_algorithm()

#Agglomerative clustering with 3 clusters

import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
X, y = make_blobs(random_state=1)
print("Agglomerative Clustering: with 3 clusters")
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("/n")

#Hierarchical Clustering, Dendrograms
print("Hiearchical cluster assignment generated with agglomerative clustering with numbered data points")
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
mglearn.plots.plot_agglomerative()

#Illustrating make_blobs clusters with dendrogram
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
# Import the dendrogram function and the ward clustering function from SciPy
from scipy.cluster.hierarchy import dendrogram, ward
X, y = make_blobs(random_state=0, n_samples=12)
# Apply the ward clustering to the data array X
# The SciPy ward function returns an array that specifies the distances
# bridged when performing agglomerative clustering
linkage_array = ward(X)
# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)
# marke the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")


