# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:18:24 2020

@author: Ian Saltzman 
        is728163@sju.edu
        Machine Learning: Representing Data and Engineering Features
UNIVARIATE STATISTICS
"""

#UNIVARIATE STATISTICS
import numpy as np
import matplotlib.pyplot as plt
# Generate Dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
# get deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# add noise features to the data
# the first 30 features are from the dataset, the next 50 are noise
X_w_noise = np.hstack([cancer.data, noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target,
 random_state=0, test_size=.5)
# use f_classif (the default) and SelectPercentile to select 58% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))
mask = select.get_support()
print(mask)
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
from sklearn.linear_model import LogisticRegression
# transform test data
X_test_selected = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
select = SelectFromModel(rfc, threshold='median')
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))
mask = select.get_support()
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
X_test_l1 = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_l1, y_train)
print("Test Score: {:.3f}".format(lr.score(X_test_l1, y_test)))
from sklearn.feature_selection import RFE
select = RFE(rfc, n_features_to_select = 40)
select.fit(X_train, y_train)
mask = select.get_support()
# visualize the mask -- black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
lr = LogisticRegression()
lr.fit(X_train_rfe, y_train)
print("Test Score using Logistics Regression: {:.3f}".format(lr.score(X_test_rfe, y_test)))
print("Test score uisng RFE: {:.3f}".format(select.score(X_test, y_test)))
import mglearn as mglearn
mglearn.plots.plot_cross_validation()

#CROSS-validation in scikit-learn on iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target)
print("Cross-validation scores: {}".format(scores))
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
import mglearn as mglearn
from sklearn.datasets import load_iris
iris = load_iris()
print("Iris labels:\n{}".format(iris.target))
mglearn.plots.plot_stratified_cross_validation()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
logreg = LogisticRegression()
kfold = KFold(n_splits=5)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("Cross-validation scores: \n{}".format(scores))
kfold = KFold(n_splits=3)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("Cross-validation scores: \n{}".format(scores))
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("Cross-validation scores: \n{}".format(scores))
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
import mglearn as mglearn
mglearn.plots.plot_shuffle_split()
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size =.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores: \n{}".format(scores))
import mglearn as mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
# Create synthetic datasets
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
logreg = LogisticRegression()
groupkfold = GroupKFold(n_splits=3)
scores = cross_val_score(logreg, X, y, groups, cv=groupkfold)
print("Cross-validation scores: \n{}".format(scores))
mglearn.plots.plot_group_kfold()

#GRID SEARCH
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
 random_state=0)
print("Size of training set: {}".format(
 X_train.shape[0], X_test.shape[0]))
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
 for C in [0.001, 0.01, 0.1, 1, 10, 100]:
     # for each combination of parameters, train an SVC
     svm = SVC(gamma=gamma, C=C)
     svm.fit(X_train, y_train)
     # evaluate the SVC on the test set)
     score = svm.score(X_test, y_test)
     # if we got a better score, stor ethe score and parameters
     if score > best_score:
         best_score = score
         best_parameters = {'C':C, 'gamma':gamma}
print("Best score: {:.2f}".format(best_score))
print("best parameters: {}".format(best_parameters))
import mglearn as mglearn
mglearn.plots.plot_threefold_split()

#THE DANGER OF OVERFITTING THE PARAMETERS AND THE VALIDATION SET
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
# split the data into train + validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
# split the train + validation set into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)
print("Size of training set: {} size of validation set: {} size of tets set:"
 "{}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        # evaluate the SVC on the validation set)
        score = svm.score(X_valid, y_valid)
        # if we got a better score, stor ethe score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
# rebuild a model on the combined training and validation set,
# and evalaute it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("Best score: {:.2f}".format(best_score))
print("best parameters: {}".format(best_parameters))
print("Test set score with best parameters: {:.2f}".format(test_score))

#GRID SEARCH WITH CROSS-VALIDATION
import mglearn as mglearn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# split the data into train + validation set and test set
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target,
 random_state=0)
# split the train + validation set into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval,
 random_state=1)
print("Size of training set: {} size of validation set: {} size of tets set:"
 "{}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
best_score=0
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C)
        #perform cross-validation
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        # compute mean cross-validation accuracy
        score = np.mean(scores)
        # if we got a better score, stor ethe score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C':C, 'gamma':gamma}
# rebuild a model on the combined training and validation set,
# and evalaute it on the test set
svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
mglearn.plots.plot_cross_val_selection()
mglearn.plots.plot_grid_search_overview()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
 random_state=0)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score {:.2f}".format(test_score))
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
print("Parameter grid:\n{}".format(param_grid))
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
 random_state=0)
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)
print("Test set score {:.2f}".format(test_score))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))
import pandas as pd
#convert to DataFrame
results = pd.DataFrame(grid_search.cv_results_)
#results.head(5)
display(results.head())
scores = np.array(results.mean_test_score).reshape(6, 6)
# plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
 ylabel='C', yticklabels=param_grid['C'], cmap='viridis')
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
param_grid_linear = {'C': np.linspace(1, 2, 6),
 'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.logspace(-3, 2, 6),
 'gamma':np.linspace(1, 2, 6)}
param_grid_range = {'C':np.logspace(-3, 2, 6),
 'gamma': np.logspace(-7, -2, 6)}
for param_grid, ax in zip([param_grid_linear, param_grid_one_log,
 param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores =grid_search.cv_results_['mean_test_score'].reshape(6, 6)

    # plot the mean cross-validation scores
    scores_image = mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
    ylabel='C', yticklabels=param_grid['C'], cmap='viridis', ax=ax)

plt.colorbar(scores_image, ax=axes.tolist())
results = pd.DataFrame(grid_search.cv_results_)
# we display the transposed table so that it better fits on the page: 
display(results.T)
param_grid = [{'kernel': ['rbf'],
 'C': [0.001, 0.01, 0.1, 1, 10, 100],
 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]},
 {'kernel': ['linear'],
 'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
print("List of grids:\n{}".format(param_grid))
grid_search = GridSearchCV(SVC(), param_grid, return_train_score= True, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
 outer_scores = []
 # for each split of the data in the outer cross-validation
 # (split method returns indices of training and test parts)
 for training_samples, test_samples in outer_cv.split(X, y):
     # find best parameter uisng inner cross-validation
     best_params = {}
     best_score = -np.inf
     # iterate over parameters
     for parameters in parameter_grid:
         # accumulate score over inner splits
         cv_scores = []
         # iterate over inner cross-validation
 for inner_train, inner_test in inner_cv.split(
         X[training_samples], y[training_samples]):
     # build classifier given parameters and training data
     clf = Classifier(**parameters)
     clf.fit(X[inner_train], y[inner_train])
     # evaluate on inner test set
     score = clf.score(X[inner_test], y[inner_test])
     cv_scores.append(score)
     # compute mean score over inner folds
 mean_score = np.mean(cv_scores)
 if mean_score > best_score:
     # if better than so far, remember parameters
     best_score = mean_score
     best_params = parameters
 # build classifier on best parameters using our training set
 clf = Classifier(**best_params)
 clf.fit(X[training_samples], y[training_samples])
 # evaluate
 outer_scores.append(clf.score(X[test_samples], y[test_samples]))
 return np.array(outer_scores)
from sklearn.model_selection import ParameterGrid, StratifiedKFold
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),
 StratifiedKFold(5), SVC, ParameterGrid(param_grid))
print("Cross-validation scores: {}".format(scores))