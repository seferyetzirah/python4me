# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:20:58 2020

@author: sefir
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
print("\n")


df= pd.read_csv('cadataexcelCSV.csv')
print(df)
print("\n")
print("Scatterplot of Total rooms & Median House Value\n")
x=df['total rooms']
y=df['median house value']
plt.scatter(x, y)
plt.show()
