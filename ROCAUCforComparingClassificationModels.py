# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:21:07 2020

@author: sefir
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import roc_curve, auc
df = pd.read_csv('diabetes.csv')
print(df)
X = df.values[:,0:8]

Y = df.values[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=1)
clf1 = LogisticRegression()

clf2 = svm.SVC(kernel='linear', probability=True)

clf3 = RandomForestClassifier()

clf4 = DecisionTreeClassifier()
probas1_ = clf1.fit(X_train, y_train).predict_proba(X_test)

probas2_ = clf2.fit(X_train, y_train).predict_proba(X_test)
probas3_ = clf3.fit(X_train, y_train).predict_proba(X_test)

probas4_ = clf4.fit(X_train, y_train).predict_proba(X_test)

fp1, tp1, thresholds1 = roc_curve(y_test, probas1_[:, 1])
roc_auc_model1 = auc(fp1, tp1)
fp2, tp2, thresholds2 = roc_curve(y_test, probas2_[:, 1])
roc_auc_model2 = auc(fp2, tp2)
fp3, tp3, thresholds3 = roc_curve(y_test, probas3_[:, 1])
roc_auc_model3 = auc(fp3, tp3)
fp4, tp4, thresholds4 = roc_curve(y_test, probas4_[:, 1])
roc_auc_model4 = auc(fp4, tp4)
print("AUC for Logistic Regression Model : ",roc_auc_model1)
print("AUC for SVM Model:", roc_auc_model2)
print("AUC for Random Forest Model :" ,roc_auc_model3)
print("AUC for Decision Tree model :", roc_auc_model4)

#Plot AUC-ROC
fpr1, tpr1, threshold1 = roc_curve(y_test, probas1_[:, 1])
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, threshold2 = roc_curve(y_test, probas2_[:, 1])
roc_auc2 = auc(fpr2, tpr2)
fpr3, tpr3, threshold3 = roc_curve(y_test, probas3_[:, 1])
roc_auc3 = auc(fpr3, tpr3)
fpr4, tpr4, threshold1 = roc_curve(y_test, probas4_[:, 1])
roc_auc4 = auc(fpr4, tpr4)

plt.clf()
plt.plot(fpr1, tpr1, label='Logistic Model (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, label='SVC Model (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, label='Random Forest Model (area = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, label='Decision Tree Model (area = %0.2f)' % roc_auc4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiverrating characteristic example')
plt.legend(loc="lower right")
plt.show()




