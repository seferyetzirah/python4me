# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:59:36 2020


@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""
#Unsupervised learning and PCA are great for image data
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
 ax.imshow(image)
 ax.set_title(people.target_names[target])
 
print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))
# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end=' ')
if (i+1) % 3 == 0:
    print()
    
#KNN of the Faces Data
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
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print()
print("K Nearest Neighbor Result")
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

#PCA of the Faces Data
import mglearn as mglearn
mglearn.plots.plot_pca_whitening()
from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))

#KNN and PCA
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print()
print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test_pca, y_test)))
print("pca.component_.shape: {}".format(pca.components_.shape))

#mORE KNN, PCA
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
 ax.imshow(component.reshape(image_shape), cmap='viridis')
 ax.set_title("{}. component".format((i+1)))
 
#It is important to do an inverse Transform after the PCA
mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

#Two PCAs
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

