#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing wrapping function
import ensemble_wrapping_function

# Importing libraries for property tests
import pandas as pd


# In[3]:


test_data = 'data/test_file.csv'
# In[7]:


def test_average_ensemble_1():
    '''
    Tests to see if pandas dataframe is returned as predictions_df
    '''
    df, csv = ensemble_wrapping_function.average_ensemble(test_data)
    assert isinstance(df, pd.DataFrame), (
        "Returned dataframe is not a Pandas Dataframe")

    return


# In[8]:


def test_average_ensemble_2():
    '''
    Tests to see if second return of csv file is
    expected NoneType (since it is a downloaded file)
    '''
    df, csv = ensemble_wrapping_function.average_ensemble(test_data)
    assert isinstance(csv, type(None)), (
        "CSV file was returned as not a NoneType, check return code")

    return
