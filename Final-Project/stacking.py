#!/usr/bin/env python
# coding: utf-8

# In[13]:
# For cleaning data
from clean_split_data import clean_data


# # Pandas library for the pandas dataframes and maths
import sys
import pandas as pd


# Import Scikit-Learn library for models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def main():
    """
    Main function to read terminal input
    """
    system_argumets = sys.argv
    try:
        if system_argumets[1] == "-stacking":
            val = input(
                "Enter your data file name",
                "(noted: must be a csv file organized with 7 features): ")
            print("starting the training process,",
                  "please give it some more time :) ")
            please_predict_me(val)
    except Exception as e:
        print(str(e))
        print("Either illegal arguments or no arguments",
              "were given by the User. Please read Reamde file")

    return


# Parses input data to appropriate format
def parsed_input_csv(data):
    """
    Converts datafile to a list
    """
    data = pd.read_csv(data)
    parsed_data = data.values.tolist()

    return parsed_data


# Predicts diagnosis based on input data
def stacking_predictor(row):
    """
    Training stacking model with our data
    Define what our base layer will be composed of and then build
    a stacking classifier base
    on these models.
    set our final estimator as "logistic regression"

    """
    our_trained_data = pd.read_csv("data/data.csv")
    our_trained_data = clean_data(our_trained_data)

    x = our_trained_data[[
        'radius_mean', 'texture_mean',
        'area_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean',
        'smoothness_mean']]
    y = our_trained_data[['diagnosis']]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()
    flattened_y_train = []
    for sub_list in y_train:
        for val in sub_list:
            flattened_y_train.append(val)

    X, y = x_train, flattened_y_train

    estimators = [
                 (
                     'random_forest', RandomForestClassifier(
                         n_estimators=5, random_state=42)),
                 ('logistic_regr', LogisticRegression(
                     solver="lbfgs", max_iter=1460)),
                 ('knn', KNeighborsClassifier(n_neighbors=5)),
                 ('svm_rbf', SVC(kernel='rbf', gamma=4, C=10000))
                 ]

    Stacking_classifier = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression(), cv=5)

    # Fit the stacking model with our own data and with selected 7 features.
    Stacking_classifier.fit(X, y)

    # Now predicting one patient
    single_predicted_result = Stacking_classifier.predict([row])

    return('%s %d' % ("patient", single_predicted_result))


# Creates dictionary of patient with boolean prediction of diagnosis
def please_predict_me(data):
    '''
    Takes parsed dataset and returns predictions
    for each patient as a dictionary. Output is displayed on main.
    '''
    parsed_data = parsed_input_csv(data)
    all_patients_result = []

    for row in parsed_data:
        individual_result = stacking_predictor(row)
        all_patients_result.append(individual_result)

    result_dict = {}
    for i, item in enumerate(all_patients_result):
        patient, classification = item.split(' ')
        patient = patient + str(i)
        print(f'{patient} is classified under class {classification}')
        result_dict[patient] = classification

    return result_dict


if __name__ == "__main__":
    main()
