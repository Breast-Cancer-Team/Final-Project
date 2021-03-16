# Import cleaning and splitting functions
from clean_split_data import clean_data
from clean_split_data import split_data


# Pandas library for the pandas dataframes
import pandas as pd


# Import Scikit-Learn library for decision tree models
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_regression, SequentialFeatureSelector
from sklearn.model_selection import train_test_split

# Import plotting libraries
import seaborn as sns
import matplotlib

# Set larger fontsize for all plots
matplotlib.rcParams.update({'font.size': 18})


# ### Data
data = pd.read_csv('data/data.csv')
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)

# ### Classifier
clf = RandomForestClassifier(n_estimators=55, random_state=42)
clf.fit(X_train, y_train)


# ###Optimized Random Forest Classifier
def feature_names():
    '''
    Returns array of input features of
    best performing backwards stepwise selection test.
    '''

    return ['radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean',
            'compactness_mean', 'concavity_mean',

            'concave points_mean', 'symmetry_mean',
            'fractal_dimension_mean']


# User input for diagnosis
def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=55, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(test_data)

    return y_pred
