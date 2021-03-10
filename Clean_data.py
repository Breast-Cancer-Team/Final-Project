#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import pandas as pd 


# In[12]:


def clean_the_data(data): 
    data=data.drop(["id","Unnamed: 32"],axis=1)
    data["diagnosis"]=data["diagnosis"].map({'B':0,'M':1}).astype(int)
    
    return data 




