#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing models

import bagging_decision_trees
import decision_trees
import gradient_boosting
import knn
import logistic_regression
import random_forest
import svm_linear
import svm_rbf


# Import cleaning and splitting
import clean_split_data


# Importing libraries for property tests
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('Final-Project/data/data.csv')


# In[3]:


data = clean_split_data.clean_data(data)
X_train, X_test, y_train, y_test = clean_split_data.split_data(data)


# In[4]:


list_of_models = [bagging_decision_trees, decision_trees, gradient_boosting,
                  knn, logistic_regression, random_forest, svm_linear, svm_rbf]


# In[5]:


def test_feature_names_1():
    '''
    Tests that the feature_names function returns a list
    '''
    for model in list_of_models:
        features = model.feature_names()
        assert isinstance(features, list), "Feature names must be in a list"
    
    return


# In[6]:


def test_feature_names_2():
    '''
    Tests that feature_names content are strings
    '''
    for model in list_of_models:
        features = model.feature_names()
        for name in features:
            assert isinstance(name, str), "Feature names must be strings"
    
    return


# In[7]:


def test_predict_1():
    '''
    Tests the returned predictions are in an array
    '''
    for model in list_of_models:
        X_train, X_test, y_train, y_test = clean_split_data.split_data(data)
        X_test = X_test[model.feature_names()]
        y_pred = model.predict(X_test)
        assert isinstance(y_pred, np.ndarray), "Predicted boolean outputs must be in an array"
    
    return


# In[8]:


def test_predict_2():
    '''
    Tests that predictions are boolean 0 or 1 values
    '''
    for model in list_of_models:
        X_train, X_test, y_train, y_test = clean_split_data.split_data(data)
        X_test = X_test[model.feature_names()]
        y_pred = model.predict(X_test)
        for pred in y_pred:
            assert isinstance(pred, np.int64)
    
    return

