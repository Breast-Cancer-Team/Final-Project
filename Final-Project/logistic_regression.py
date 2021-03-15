#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import pandas and plotting libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Import Scikit-Learn library for the regression models and confusion matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Import cleaning and splitting functions
from clean_split_data import clean_data
from clean_split_data import split_data


# ### Data
data = pd.read_csv('data/data.csv')
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)


# ### Classifier
clf = LogisticRegression(solver="saga", max_iter=5000)
clf.fit(X_train, y_train)


# ### Optimized Logistic Regression Predictor
def feature_names():
    '''
    Returns array of input features of
    best performing backwards stepwise selection test.
    '''

    return ['radius_mean',
            'texture_mean',
            'perimeter_mean',
            'area_mean',
            'smoothness_mean',
            'compactness_mean',
            'concavity_mean',
            'concave points_mean',
            'symmetry_mean',
            'fractal_dimension_mean']


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    logistic_reg = LogisticRegression(solver="saga", max_iter=5000)
    logistic_reg.fit(X_train, y_train)
    y_pred = logistic_reg.predict(test_data)

    return y_pred
