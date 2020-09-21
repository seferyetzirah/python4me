# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:43:34 2020

@author: sefir
"""

#MLPs Multilayer Perceptrons “Neural Networks”
import mglearn as mglearn
# Visualization of logistics regression
display(mglearn.plots.plot_logistic_regression_graph())
# Visualization of MLP
display(mglearn.plots.plot_single_hidden_layer_graph())

#Relu and Tanh nonlinear functions
import mglearn as mglearn
import matplotlib.pyplot as plt
import numpy as np
line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")
plt.plot(line, np.maximum(line, 0), label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")


