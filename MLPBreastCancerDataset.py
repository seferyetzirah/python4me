# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:10:32 2020

@author: sefir
"""
#Neural Network MLP on Breast Cancer Dataset
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Breast Cancer Data Set default MLP ")
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
print("\n")

#Scaling with StandardScaler
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
#subtract the mean, and scale by inversse standard deviation
#afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Breast Cancer Data Set Scaled MLP ")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
print("\n")

#Increase the number of iterations
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
#subtract the mean, and scale by inversse standard deviation
#afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Breast Cancer Data Set Increase Iterations MLP ")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))    
print("\n")

#SET ALPHA = 1
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
#subtract the mean, and scale by inversse standard deviation
#afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(max_iter=1000, alpha = 1, random_state=42)
mlp.fit(X_train_scaled, y_train)
print("Breast Cancer Data Set Alpha to 1 MLP ")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
print("\n")

#Looking at the weights in the model
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
training_accuracy = []
test_accuracy = []
# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set
std_on_train = X_train.std(axis=0)
#subtract the mean, and scale by inversse standard deviation
#afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train
#use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

