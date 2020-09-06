# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:32:49 2020

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
plt.plot(x, y, marker = 'x') 

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



