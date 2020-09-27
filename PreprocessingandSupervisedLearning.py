# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 20:25:51 2020

@author: is721863@sju.edu Ian Saltzman
Unsupervised  Learning
"""

#Preprocessing on Supervised Learning
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
stratify=cancer.target, random_state=66)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler
#preprocessing using 0-1 scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#learn an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)
#scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler
#preprocessing using 0-1 scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#learn an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)
#scoring on the scaled test set
print("SVM test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))

