# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:36:04 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
Kmeans, DBSCAN, Agglomerative clustering comparison with the
Faces in the Wild data set
"""

# Generate Dataset
#Faces in the wild initial algo
import mglearn as mglearn
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

image_shape = people.images[0].shape
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names

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
# build a KNeighborsClassifier using one neighbor
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels)))

#Comparing what DBSCAN finds as noise to other analysis methods

dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels)))
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels)))
# Count number of points in all clusters and noise.
# bincount doesn't allow negative numbers so we need to add 1.
# The first number in the result corresponds to noise points.
print("Number of points per cluster: {}".format(np.bincount(labels+1)))
noise = X_people[labels==-1]
fig, axes = plt.subplots(3, 9, figsize=(12, 4), subplot_kw={'xticks': (), 'yticks': ()})
for image, ax in zip(noise, axes.ravel()):
 ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
for eps in [1, 3, 5, 7, 9, 11, 13]:
 print("\neps={}".format(eps))
 dbscan = DBSCAN(min_samples=3, eps=eps)
 labels = dbscan.fit_predict(X_pca)
 print("Number of clusters: {}".format(len(np.unique(labels))))
 print("Cluster sizes: {}".format(np.bincount(labels+1)))
 
 
 #focus on the eps=7
 dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)
for cluster in range(max(labels)+1):
 mask=labels==cluster
 n_images = np.sum(mask)
 fig, axes = plt.subplots(1, n_images, figsize=(n_images*1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
 for image, label, ax in zip(X_people[mask], y_people[mask], axes):
     ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
     ax.set_title(people.target_names[label].split()[-1])
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print("Unique labels: {}".format(np.unique(labels)))
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# extract clusters with k-means
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))
fig, axes = plt.subplots(2, 5, figsize=(12, 4), subplot_kw={'xticks': (), 'yticks': ()})
for center, ax in zip(km.cluster_centers_, axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people,
people.target_names)
# extract clusters with ward agglomerative clustering
from scipy.cluster.hierarchy import dendrogram, ward
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("Cluster sizrs agglomerative clustering: {}".format(np.bincount(labels_agg)))
from sklearn.metrics.cluster import adjusted_rand_score
print("ARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))
linkage_array = ward(X_pca)
# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters