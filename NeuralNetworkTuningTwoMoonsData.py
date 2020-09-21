# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:49:01 2020

@author: sefir
"""
#NOTE EACH # DENOTES A DIFFERENT MODEL ADJUSTMENT, SEPERATE INTO SEPERATE CODING CONSOLES
#FOR USE IN APPLICATIONS FOR ML
#Tuning neural networks -- with the Two Moons Data
import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


#Reduce Hidden Layer size
import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10]).fit(X_train,
y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#2 hiden layers size 10, relu
import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
# using two hidden layers, with 10 units each
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10]).fit(X_train,
y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#2 hidden layers 10, tanh
#tanh and 2 hidden layers 
import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
# using two hidden layers, with 10 units each
mlp = MLPClassifier(solver='lbfgs', activation ='tanh', random_state=0,
hidden_layer_sizes=[10, 10]).fit(X_train, y_train)

mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

#L2 Alpha Penalty Regularizartion to compare performance

import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0,
                            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train, ax=ax)
ax.set_title("n_hidden=[{}, {}]\n alpha={:.4f}".format(
n_hidden_nodes, n_hidden_nodes, alpha))
#random weights
import mglearn as mglearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# split the wave dataset into training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    mlp = MLPClassifier(solver='lbfgs', random_state=i,
                        hidden_layer_sizes=[100, 100])
mlp.fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
mglearn.discrete_scatter(X_train[:, 0],X_train[:, 1], y_train, ax=ax)


