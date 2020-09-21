# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:09:07 2020

@author: sefir
"""

#uncertainty in multiclass classification
import numpy as np
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
iris=datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
# build the gradient boosting model
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
# Show the first few entires of decision_function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
print("Argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Predictions: \n{}".format(gbrt.predict(X_test)))
# Show the first few entires of predict_proba
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6, :]))
# Show that sums across rows are one
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
print("Argmax of predicted probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("Predictions: \n{}".format(gbrt.predict(X_test)))
