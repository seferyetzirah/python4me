# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:48:46 2020

@author: Ian Saltzman 
        is728163@sju.edu
        Machine Learning: Representing Data and Engineering Features
        
"""
# HANDLING CATEGORICAL DATA – CREATING AN EXAMPLE DATASET
import pandas as pd
df = pd.DataFrame([
 ['green', 'M', 10.1, 'class1'],
 ['red', 'L', 13.5, 'class2'],
 ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
df

#Mapping Original Features
size_mapping = {
 'XL' : 3,
 'L' :2,
 'M': 1}
df['size'] = df['size'].map(size_mapping)
df
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)

# ENCODING CLASS LABELS
import numpy as np
class_mapping = {label:idx for idx, label in
 enumerate(np.unique(df['classlabel']))}
class_mapping
df['classlabel']=df['classlabel'].map(class_mapping)
df
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y =class_le.fit_transform(df['classlabel'].values)
y
class_le.inverse_transform(y)

# PERFORMING ONE-HOT ENCODING ON NOMINAL FEATURES
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

# PERFORMING ONE-HOT ENCODING ON NOMINAL FEATURES
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer(
 [('onehot', OneHotEncoder(), [0]), # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
 ('nothing', 'passthrough', [1,2]) ] # Leave the rest of the columns untouched
)
c_transf.fit_transform(X).astype(float)
pd.get_dummies(df[['price', 'color', 'size']])
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer(
 [('onehot', color_ohe, [0]), # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
 ('nothing', 'passthrough', [1, 2]) # Leave the rest of the columns untouched
])
c_transf.fit_transform(X).astype(float)

# ONE-HOT-ENCODING (DUMMY VARIABLES)
import mglearn as mglearn
import pandas as pd
import os
# This file has no headers naming the columns, so we pass header=None
# and provide the column names explicitly to "names"
adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(adult_path, header=None, index_col=False,
 names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
 'marital status', 'occupation', 'relationship', 'race', 'gender',
 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
 'income'])
# For illustration purposes, we only select some of the columns
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
 'occupation', 'income']]
# IPython display allows nice output formatting within the Spyder
display(data.head())
print(data.gender.value_counts())
print(data.gender.value_counts())
print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))
data_dummies.head()
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
# Extract NumPy arrays
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

# Applying Logistics Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
 'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
display(demo_df)
display(pd.get_dummies(demo_df))
# Explicitly listing the columns to encode
demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
demo_df.style.format("{:.2%}")
pd.set_option('display.max_columns', 1000) # or 1000
pd.set_option('display.max_rows', None) # or 1000
pd.set_option('display.max_colwidth', 1) # or 199
display(pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature']))


