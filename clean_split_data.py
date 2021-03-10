import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(data):
    '''
    Drops ID column of data as well as Not a Number column at end of dataset.
    Retains all mean measurement columns and drops all standard errors and worst measurements.
    Remaps diagnosis column to numbered booleans.
    '''
    data.drop(list(data.filter(regex='_se')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='_worst')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='id')), axis=1, inplace=True)
    data.drop(list(data.filter(regex='Unnamed: 32')), axis=1, inplace=True)
    data['diagnosis'].replace('B', 0, inplace=True)
    data['diagnosis'].replace('M', 1, inplace=True)
    
    return data


def split_data(data):
    '''
    Takes cleaned dataset and splits into train, test sets.
    '''
    X = data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test




