# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:23:28 2020

@author: Ian Saltzman
is721863@sju.edu
Diabetes Database from Pima Indians from Kaggle
A simple K Nearest Neighbor

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split

df= pd.read_csv('diabetes.csv')
print(df)
print("Diabetes.keys():\n", df.keys())
print("\n")
print("Shape of Data:", df.shape)
print("\n")



from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('Outcome', axis=1))
scaled_data = pd.DataFrame(df, columns = df.drop('Outcome', axis=1).columns)

X = scaled_data
y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train) 
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
print("N of 4")
print("\n")


knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
print("N of 9 chosen after reviewing accuracy")
print("\n")


training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
print("Testing and Traning accuracy")
print("\n")

plt.plot(neighbors_settings, training_accuracy, label="Training Accuracy")
plt.plot(neighbors_settings, test_accuracy, label="Test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

