#!/usr/bin/env python
# coding: utf-8




import sys
import pandas as pd 
from numpy import mean
from numpy import std


import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from clean_split_data import clean_data
from clean_split_data import split_data


def main():
    """
    Main function to read terminal input
    """
    system_argumets = sys.argv
    try:
        if system_argumets[1] == "-stacking":
            val = input("Enter your data file name (noted: must be a csv file organized with 10 features): ")
            please_predict_me(val)
    except Exception as e:
        print(str(e))
        print("Either illegal arguments or no arguments were given by the User. Please read Reamde file")



def parsed_input_csv(data): 
    data = pd.read_csv(data)
    parsed_data = data.values.tolist()
    return parsed_data
        

def stacking_predictor(row): 
    """
    Trainning stacking model with our data 
    Define what our base layer will be composed of and then build a stacking classifier base
    on these models. 
    set our final estimator as "logistic regression"
    
    """
    our_trained_data = pd.read_csv("data.csv")
    our_trained_data = clean_data(our_trained_data)

    x=our_trained_data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
    y=our_trained_data[['diagnosis']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()



    flattened_y_train = [] 
    for sub_list in y_train: 
        for val in sub_list: 
            flattened_y_train.append(val)

    X, y = x_train, flattened_y_train
    
    estimators = [
    ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
    ('log', LogisticRegression(solver="lbfgs", max_iter=146)),
    ('knn', KNeighborsClassifier(n_neighbors =5)),
    ('svm', SVC(kernel='rbf', gamma=4, C=10000))
]
    Stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv = 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    Stacking_classifier.fit(X_train, y_train)
    
    #Now predicting one patient 
    single_predicted_result = Stacking_classifier.predict([row])
#     single_probability = Stacking_classifier.predict_proba([row])
    
    return('%s %d' % ("patient", single_predicted_result))
    
def please_predict_me(data):

    parsed_data = parsed_input_csv(data)
    All_patients_result = [] 
    
    for row in parsed_data:
        Individual_result = stacking_predictor(row)
        All_patients_result.append(Individual_result)
        
    return str(All_patients_result)[1:-1]
    print(str(All_patients_result)[1:-1])

def asking_for_input(val): 

    val = input("Enter your data file name (noted: must be a csv file organized with 10 features): ") 
    please_predict_me(val)
    
    
if __name__ == "__main__":
    main()





