#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import cleaning and splitting
import clean_split_data

# Importing libraries for property tests
import math
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('data.csv')


# In[3]:


def test_clean_data_1():
    '''
    Test to determine if data column contains all strings
    '''
    df = clean_split_data.clean_data(data)
    names = df.columns
    for name in names:
        assert isinstance(name, str)
    
    return


# In[4]:


def test_clean_data_2():
    '''
    Test to determine if any of the data contains strings "_se"
    '''
    df = clean_split_data.clean_data(data)
    names = df.columns
    substring = "_se"
    for name in names:
        assert substring not in name, "Standard error columns still exist in dataframe"
    
    return


# In[5]:


def test_clean_data_3():
    '''
    Test to determine if any of the data contains strings "_worst"
    '''
    df = clean_split_data.clean_data(data)
    names = df.columns
    substring = "_worst"
    for name in names:
        assert substring not in name, "Worst measurement columns still exist in dataframe"
    
    return


# In[6]:


def test_clean_data_4():
    '''
    Test to determine if "id" column was successfully dropped from dataframe
    '''
    df = clean_split_data.clean_data(data)
    names = df.columns
    substring = "id"
    for name in names:
        assert substring not in name, "ID column still exists in dataframe"
    
    return


# In[7]:


def test_clean_data_5():
    '''
    Test to determine if "Unnamed: 32" column was successfully dropped from dataframe
    '''
    df = clean_split_data.clean_data(data)
    names = df.columns
    substring = "Unnamed: 32"
    for name in names:
        assert substring not in name, "Unnamed: 32 column still exists in dataframe"
    
    return


# In[8]:


def test_clean_data_6():
    '''
    Test to determine if diagnosis values are replaced with integers
    '''
    df = clean_split_data.clean_data(data)
    diagnosis = df.diagnosis
    for cancer in diagnosis:
        assert isinstance(cancer, int), "Diagnosis values are not integers"
    
    return


# In[9]:


def test_split_data_1():
    '''
    Test to determine total length of datafile did not change when splitting
    '''
    total_length = len(data)
    df = clean_split_data.clean_data(data)
    X_train, X_test, y_train, y_test = clean_split_data.split_data(df)
    train_length = len(X_train)
    test_length = len(X_test)
    total_split = train_length + test_length
    assert math.isclose(total_length, total_split), "Length of data is not the same as before splitting"
    
    return


# In[10]:


def test_split_data_2():
    '''
    Test to determine proportion of split is correct
    '''
    total_length = len(data)
    df = clean_split_data.clean_data(data)
    X_train, X_test, y_train, y_test = clean_split_data.split_data(df)
    train_length = len(X_train)
    train_split = train_length / total_length
    assert math.isclose(train_split, 0.80, abs_tol=0.1), "Training set is not at specified 80% of dataset"
    
    return

