#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing stacking ensemble
import stacking


# Importing libraries for property tests
import math
import numpy as np
import pandas as pd


# In[2]:


file_name = 'Final-Project/data/data2.csv'


# In[3]:


def test_parsed_input_csv_1():
    '''
    Test to determine if returned output is a list
    '''
    parsed_data = stacking.parsed_input_csv(file_name)
    assert isinstance(parsed_data, list), "Parsed data is not a type list. There is a cleaning issue"
    
    return


# In[4]:


def test_parsed_input_csv_2():
    '''
    Test to determine if returned output is a list of lists
    '''
    parsed_data = stacking.parsed_input_csv(file_name)
    for data in parsed_data:
        assert isinstance(data, list), "Parsed data is not a type list. There is a cleaning issue"
    
    return


# In[5]:


def test_stacking_predictor_1():
    '''
    Test to determine if returned output is a string with "patient" and the diagnosis "0" or "1" concatenated
    '''
    parsed_data = stacking.parsed_input_csv(file_name)
    rand_row = parsed_data[1]
    result = stacking.stacking_predictor(rand_row)
    assert isinstance(result, str), 'Resulting output is not a concatenated string'
    
    return


# In[6]:


def test_stacking_predictor_2():
    '''
    Test to determine if first word of returned output is patient
    '''
    parsed_data = stacking.parsed_input_csv(file_name)
    rand_row = parsed_data[1]
    result = stacking.stacking_predictor(rand_row)
    assert result.split()[0] == 'patient', 'First string in output is not "patient"'
    
    return


# In[7]:


def test_stacking_predictor_3():
    '''
    Test to determine if second word of returned output is 0 or 1
    '''
    parsed_data = stacking.parsed_input_csv(file_name)
    rand_row = parsed_data[1]
    result = stacking.stacking_predictor(rand_row)
    assert result.split()[0] == '0' or result.split()[1] == '1', 'Diagnosis is neither boolean 0 or 1'
    
    return


# In[8]:


def test_please_predict_me_1():
    '''
    Test to determine if returned output is a dictionary
    '''
    result_dict = stacking.please_predict_me(file_name)
    assert isinstance(result_dict, dict)
    
    return


# In[9]:


def test_please_predict_me_2():
    '''
    Test to determine if diagnosis in dictionary is a string
    '''
    result_dict = stacking.please_predict_me(file_name)
    for key in result_dict:
        assert isinstance(result_dict[key], str), 'Daignosis is not string at end of dictionary value'
    
    return


# In[10]:


def test_please_predict_me_3():
    '''
    Test to determine if diagnosis in dictionary is a string 0 or 1
    '''
    result_dict = stacking.please_predict_me(file_name)
    for key in result_dict:
        assert result_dict[key] == '0' or result_dict[key] == '1', 'Diagnosis is neither 0 or 1 string values'
    
    return

