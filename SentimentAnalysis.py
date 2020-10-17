# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 12:47:19 2020

@author: Ian Saltzman is721863@sju.edu
Predictive Analytics: Machine Learning
Sentiment Analysis: Text Mining

"""
#EXAMPLE APPLICATION: SENTIMENT ANALYSIS OF MOVIE REVIEWS 
from sklearn.datasets import load_files
import numpy as np
reviews_train = load_files("aclImdb/train/")
                           
text_train, y_train = reviews_train.data, reviews_train.target

print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))
# Cleaning the data and removing the formatting before we proceed

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

print("Samples per class (training): {}".format(np.bincount(y_train)))
reviews_test = load_files("aclImdb/train/")
text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

#N. APPLYING BAG-OF-WORDS TO A TOY DATASET

bards_words = ["The fool doth think he is wise,",
 "but the wise man knows himself to be a fool"]
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(bards_words)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))
# Creating bag of words
bag_of_words = vect.transform(bards_words)
print("bag_of_words: {}".format(repr(bag_of_words)))
# Converting the sparse matrix into dense matrix
print("Dense representation of bag_of_words:\n{}".format(
 bag_of_words.toarray()))

#BAG-OF-WORDS FOR MOVIE REVIEWS
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(text_train)
X_train = vect.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Fetaures 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th feature: \n{}".format(feature_names[::2000]))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Log Reg Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
X_test = vect.transform(text_test)
print("test score: {:.2f}".format(grid.score(X_test, y_test)))
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with min_df: {}".format(repr(X_train)))
feature_names = vect.get_feature_names()
print("First 50 features:\n{}".format(feature_names[:50]))
print("Fetaures 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 700th feature:\n{}".format(feature_names[::700]))
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Grid seacrcb 2 Best cross-validation score: {:.2f}".format(grid.best_score_))
# Stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stopword: \n{}".format(list(ENGLISH_STOP_WORDS)[::10]))
# Specifying stop_words="english" uses the buil-in list
# We could also augment it and pass our own
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words:\n{}".format(repr(X_train)))
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Stopword Grid Search Best cross-validation score: {:.2f}".format(grid.best_score_))

#Tf-idf rescaling

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(TfidfVectorizer(min_df=5),
 LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print(" Tf idf Best cross-validation score: {:.2f}".format(grid.best_score_))
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_train = vectorizer.transform(text_train)
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorizer.get_feature_names())
print("Feature with lowest tfidf:\n{}".format(
 feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(
 feature_names[sorted_by_tfidf[:-20]]))
sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".format(
 feature_names[sorted_by_idf[:100]]))

# Investigating Model Coefficients

import mglearn as mglearn
pipe = make_pipeline(TfidfVectorizer(min_df=5),
 LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
from sklearn.model_selection import GridSearchCV
mglearn.tools.visualize_coefficients( grid.best_estimator_.named_steps["logisticregression"].coef_,
 feature_names, n_top_features=40)
# Using N-grams
print("bards_words: \n{}".format(bards_words))
cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
print("Transformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
pipe = make_pipeline(TfidfVectorizer(min_df=5),
 LogisticRegression())
# running the grid search takes a long time because of the
# relatively large grid and the inclusion of triggers
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
 "tfidfvectorizer__ngram_range":[(1,1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: \n{}".format(grid.best_params_))
# extract scores from grid_search
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
# visualize heatmap
import matplotlib.pyplot as plt
heatmap = mglearn.tools.heatmap(
 scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
 xticklabels=param_grid['logisticregression__C'],
 yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
# extract feature names and coefficients
vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
# find 3-gram features
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
# visualize only 3-gram features
mglearn.tools.visualize_coefficients(coef.ravel()[mask],
 feature_names[mask], n_top_features=40)