#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pandas library for the pandas dataframes
import pandas as pd    
import numpy as np

# Import Scikit-Learn library for decision tree models
import sklearn         
from sklearn import linear_model, datasets
from sklearn.utils import resample
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix


# Import plotting libraries
import seaborn as sns
import matplotlib 
from matplotlib import pyplot as plt


# Set larger fontsize for all plots
matplotlib.rcParams.update({'font.size': 18})
from IPython.display import clear_output


# Import cleaning and splitting functions
from clean_split_data import clean_data
from clean_split_data import split_data


# ### Data
data = pd.read_csv('data/data.csv')
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)

# ### Classifier
# Default criterion is GINI index
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_bag = BaggingClassifier(base_estimator=clf, n_estimators=43, random_state=42)
clf_bag.fit(X_train, y_train)

# ### Optimized Bagging Predictor
def feature_names():
    '''
    Returns array of input features of best performing backwards stepwise selection test.
    '''
    
    return ['texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
       'concavity_mean', 'concave points_mean', 'symmetry_mean']


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf_bag = BaggingClassifier(base_estimator=classifier, n_estimators=43, random_state=42)
    clf_bag = clf_bag.fit(X_train, y_train)
    y_predict = clf_bag.predict(test_data)
    
    return y_predict
