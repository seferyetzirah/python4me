# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:03:31 2020

@author: sefir

@author: is721863@sju.edu ian@laurelvalleyfarms.net
Regression and Correlation Coefficients for an analysis of  16-18 Stockpile to Phase 2 Spawn Cellulose.


"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


df= pd.read_csv('PreSP.CSV')
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


X=df[['PRESP.FAT', 'PRESP.WAC5', 'PRESP.WAC24', 'PRESP.RESISTANCE', 'PRESP.STRUCTURE',  'PRESP.HEMI', 'PRESP.CELLULOSE' ]].values
y=df[['PH2.CELLULOSE']].values



lr = LinearRegression().fit(X, y)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X, y)))
print("Test set score: {:.2f}".format(lr.score(X, y)))

corM = df.corr()
print("Correl Coeff: {}".format(df.corr()))

import seaborn as sns
df_small = df.iloc[:,:11]
correlation_mat = df_small.corr()
sns.heatmap(correlation_mat, annot= True)
plt.show()