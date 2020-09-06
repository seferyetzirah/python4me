# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:32:49 2020
Modified Fri Sep 4 20:33:05 2020

@author: sefir
"""

import numpy as np 
x = np.array([[1, 2, 3], [4, 5, 6]]) 
print("x:\n{}".format(x)) 
print(type(x)) 
print(x.shape) 

from scipy import sparse

#  2d numpy array w/ diagonal of 1s
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))


# convert numpy array to scipy array sparse matrix in CSR format
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix)) 

# sparse matrix in COO format
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO representation:\n{}".format(eye_coo)) 

import matplotlib.pyplot as plt
#generate seq of numbers -10 to 10 w 100 steps in between
x= np.linspace(-10, 10, 100)
#create 2nd array using sine
y = np.sin(x)
#plot to line chart one array against another
#plt.plot(x, y, marker = 'x') 

import pandas as pd

#dataset of people in pandas
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", " Berlin", "London"],
        'Age': [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
#try pretty printing
display(data_pandas)

display(data_pandas[data_pandas.Age > 30])

import sys 
print("Python version: {}".format(sys.version)) 
import pandas as pd 
 
print("pandas version: {}".format(pd.__version__)) 
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__)) 
import numpy as np 
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython 
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("Scikit-learn version: {}".format(sklearn.__version__)) 

# leaving this as a note that below is code for activity 2
print("1a \n")
import numpy as np
from sklearn.model_selection import train_test_split
# create 2d numpy array X and 1d array y
X,y = np.arange(10).reshape((5,2)), range(5)
print("X:\n{}".format(X))
print("y:\n{}".format(list(y)))
#divide X and y into train and test sets
#test size 1/3 implies 2/3 is training
#random state is 42, shuffle is true by default
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42)
print("X_train:\n{}".format(X_train))
print("y_train:\n{}".format(y_train))
print("X_test:\n{}".format(X_test))
print("y_test:\n{}".format(y_test))
#set shuffle to false
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
print("y_train:\n{}".format(y_train))
print("y_test:\n{}".format(y_test))

#2 Iris dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print dictionary keys
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193]+ "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data']))) 
("Shape of data:  {}".format(iris_dataset['data'].shape)) 
print("First five rows of data:\n{}".format(iris_dataset['data'][:5])) 
print("Type of target: {}".format(type(iris_dataset['target']))) 
print("Shape of target:  {}".format(iris_dataset['target'].shape)) 
print("Target:\n{}".format(iris_dataset['target']))
 #The meaning of the number is given by the iris_dataset['target_names'] array: 0 - setosa, 1- versicolor, 2 - virginica 
 
 #splitting into training, testing
print("Splitting data set \n")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(iris_dataset['data'], 
iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape)) 
print("y-train shape: {}".format(y_train.shape)) 
print("X_test shape: {}".format(X_test.shape)) 
print("y-test shape: {}".format(y_test.shape))

# C vizualizing the dataset
#create dataframe from data in X_train
#label columns using the strings in iris_dataset.feature_names

import pandas as pd
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
#scatter  matrix from the dataframe, color by y_train
import mglearn as mglearn

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3 )
