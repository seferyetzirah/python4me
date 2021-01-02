""""
@author: Ian Saltzman is721863@sju.edu
12/21/2020 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



df= pd.read_csv('Hay.CSV')
print(df)
print("\n")
print("Summary Statistics\n")
print(df.describe())
print("\n")
print("Sum\n")
print(df.sum())
print("\n")
print("Median\n")
print(df.median())
print("\n")
print("Mode\n")
print("\n")
print(df.mode())
print("\n")
print("\n")


X=df[['HAY.FAT', 'HAY.WAC5', 'HAY.WAC24']].values
y=df[['PH2.CELLULOSE']].values
from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X, y)
print("Training set score: {:.2f}".format(lr.score(X, y)))
print("Test set score: {:.2f}".format(lr.score(X, y))
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X, y)))
print("Test set score: {:.2f}".format(lr.score(X, y)))

