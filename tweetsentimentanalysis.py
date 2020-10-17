# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:05:52 2020

@author: sefir
"""

from sklearn.feature_extraction.text import CountVectorizer

data = []
data_labels = []
with open("C:/Users/sefir/Desktop/DSS 740 ML/7/twtw/pos_tweets.txt", encoding ="utf8") as f:
    for i in f: 
        data.append(i) 
        data_labels.append('pos')

with open("C:/Users/sefir/Desktop/DSS 740 ML/7/twtw/neg_tweets.txt", encoding ="utf8") as f:
    for i in f: 
        data.append(i)
        data_labels.append('neg')
vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)
features = vectorizer.fit_transform(
    data
)
features_nd = features.toarray() # for easy usage

from sklearn.model_selection import train_test_split
print("Selecting random tweets via classifiers pos or neg")
print("\n")
X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
import random
j = random.randint(0,len(X_test)-7)
for i in range(j,j+7):
    print(y_pred[0])
    ind = features_nd.tolist().index(X_test[i].tolist())
    print(data[ind].strip())
    
from sklearn.metrics import accuracy_score
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred)))
