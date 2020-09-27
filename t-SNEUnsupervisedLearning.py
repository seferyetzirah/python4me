# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:54:51 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""
#Manifold t-SNE Learning using Scikitlearn Handwritten Digits Data
import matplotlib.pyplot as plt
import numpy as np
import mglearn as mglearn
from sklearn.datasets import load_digits
digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
 ax.imshow(img)
print("Examples of the Digits Data set")
print("\n")

#PCA Model
# build a PCA model
print("PCA Model of Digit Data")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(digits.data)
# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A24", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683",
"#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
 # actually plot the digits as text instead of using scatter
 plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]],
fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")

#t-SNE Model

from sklearn.manifold import TSNE
print("t-SNE Model")
tsne = TSNE(random_state=42)
#use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()+1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()+1)
for i in range(len(digits.data)):
 # actually plot the digits as text instead of using scatter
 plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color = colors[digits.target[i]],
fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")

