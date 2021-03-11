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

# In[2]:


data = pd. read_csv("data.csv")


# In[3]:


data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)


# ### Classifier

# In[4]:


logistic_reg = LogisticRegression(solver="lbfgs", max_iter=146) 
logistic_reg.fit(X_train, y_train)


# ### Logistic Regression Prediction (User Input)

# In[5]:


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    test_data = pd.DataFrame(test_data).T
    y_log = logistic_reg.predict(test_data)
    
    return y_log


# ### Sample Train, Test, Split Results

# In[6]:


def sample_results():
    '''
    Returns the results and confusion matrix of the sample dataset from Breast Cancer Wisconsin Dataset.
    '''
    y_log = logistic_reg.predict(X_test)
    print("Mean accuracy on test set: ", logistic_reg.score(X_test, y_test))
    print(classification_report(y_test, y_log))
    print("The confusion matrix for the sample dataset using Logistic Regression is displayed below:")
    plot_confusion_matrix(logistic_reg, X_test, y_test)
    plt.show
    
    return

