#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas library for the pandas dataframes and other plotting tools
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Import Scikit-Learn library for models
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix # plotting confusion matrix 
from sklearn.metrics import classification_report

# Import cleaning and splitting functions
from clean_split_data import clean_data
from clean_split_data import split_data

# For confusion matrix count
from collections import Counter


# ### Data
data = pd.read_csv("data.csv")
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)

# ### Classifier
clf = KNeighborsClassifier(n_neighbors=20, weights='uniform')
clf.fit(X_train, y_train)

# ### Optimized KNN Predictor
def feature_names():
    '''
    Returns array of input features of best performing backwards stepwise selection test.
    '''
    
    return ['texture_mean', 'perimeter_mean', 'smoothness_mean',
       'compactness_mean', 'concavity_mean', 'concave points_mean',
       'symmetry_mean', 'fractal_dimension_mean']


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=20, weights='uniform')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(test_data)
    
    return y_predict
