#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing wrapping function
import ensemble_wrapping_function


# Import cleaning and splitting
import clean_split_data


# Importing libraries for property tests
import math
import numpy as np
import pandas as pd


# In[3]:


test_data = 'test_file.csv'
# In[7]:


def test_average_ensemble_1():
    '''
    Tests to see if pandas dataframe is returned as predictions_df
    '''
    df, csv = ensemble_wrapping_function.average_ensemble(test_data)
    assert isinstance(df, pd.DataFrame)
    
    return


# In[8]:


def test_average_ensemble_2():
    '''
    Tests to see if second return of csv file is expected NoneType (since it is a downloaded file)
    '''
    df, csv = ensemble_wrapping_function.average_ensemble(test_data)
    assert isinstance(csv, type(None))
    
    return

