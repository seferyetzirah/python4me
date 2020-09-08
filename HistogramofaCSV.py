# -*- coding: utf-8 -*-
"""
Standard format for MlEARNING  Aug 26 22:08:01 2020
@author: Ian Saltzman is721863@sju.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

df= pd.read_csv('cadataexcelCSV.csv')
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
print("Vizualizations of the Data")
print("\n")
print("Histogram\n")
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
df.plot(kind= 'hist', bins=9)
print("\n")




