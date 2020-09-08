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
import statistics

t = (['101', '325', '411', '52', '211', 
                          '371', '290', '573', '484', '137', ])
df = pd.DataFrame(t)

print(df)
print("\n")
print("Average\n")
print(df.mean())

print("Median\n")
print(df.median())
print("Mode\n")
statistics.mode([101, 325, 411, 52, 211, 371, 290,
                 573, 484, 137])
print("None")





