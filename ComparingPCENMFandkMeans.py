# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 22:16:16 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""

#SCIKIT-LEARN and kMeans
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# generate synthetic two-dimensional data
X, y = make_blobs(random_state=1)
# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster membership:\n{}".format(kmeans.labels_))
print("k-Means Prediction membership:\n{}".format(kmeans.predict(X)))

#k-means and clustering
# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("Cluster membership:\n{}".format(kmeans.labels_))
print("k-Means Prediction membership:\n{}".format(kmeans.predict(X)))
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1,
2], markers='^', markeredgewidth=2)

#different cluster numbers

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# using two cluster centers
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])
# using five cluster centers:
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
#issues with diameters not same

X_varied, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5],
random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
plt.figure(figsize=(8, 4))
mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#Directions in kMeans
# generate some random cluster data
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)
# transform the data to be stretched
transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)
# cluster the data into three cluster centers
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.figure(figsize=(8, 2))
# plot the cluster assignments and cluster centers
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1,
2], markers='^', markeredgewidth=2)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


#complex shapes and kmeans
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
# generate synthetic two-moons data (with less noise this time)
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# build the clustering model with two clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)
# plt.figure(figsize=(8, 2))
# plot the cluster assignments and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^',
 c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2, edgecolor='k')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#PCA, NMF, KMEANS
# Generate Dataset
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
 mask[np.where(people.target == target)[0][:50]] = 1
X_people = people.data[mask]
y_people = people.target[mask]
# scale the grayscale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255
# split the data into training in test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people,
random_state=0)
from sklearn.decomposition import NMF
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
from sklearn.decomposition import PCA
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
from sklearn.cluster import KMeans
# build the clustering model with hundred clusters
kmeans = KMeans(n_clusters=100)
kmeans.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_,
pca.components_, nmf.components_):
 ax[0].imshow(comp_kmeans.reshape(image_shape))
 ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
 ax[2].imshow(comp_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("kmeans")
axes[1, 0].set_ylabel("pca")
axes[2, 0].set_ylabel("nmf")
fig, axes = plt.subplots(4, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Reconstructions")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeans,
X_reconstructed_pca, X_reconstructed_nmf):
 ax[0].imshow(orig.reshape(image_shape))
 ax[1].imshow(rec_kmeans.reshape(image_shape))
 ax[2].imshow(rec_pca.reshape(image_shape))
 ax[3].imshow(rec_nmf.reshape(image_shape))
axes[0, 0].set_ylabel("original")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")

#variation in a complex model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
# build the clustering model with hundred clusters
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred=kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^',
 s=60, c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
print("Cluster membership:\n{}".format(y_pred))


