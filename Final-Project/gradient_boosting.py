# Data transformations
from clean_split_data import clean_data
from clean_split_data import split_data


# Pandas library for the pandas dataframes
import pandas as pd

# Import Scikit-Learn library for the regression models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Import plotting libraries
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt

# Set larger fontsize for all plots
matplotlib.rcParams.update({'font.size': 20})


# ### Data
data = pd.read_csv('data/data.csv')
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)

# ### Classifier
tree_count = 10
gradient_model = GradientBoostingClassifier(
    n_estimators=tree_count, learning_rate=0.1, max_depth=10, random_state=42)
gradient_model.fit(X_train, y_train)


# ### Optimized Gradient Boosting Predictor
def feature_names():
    '''
    Returns array of input features of best
    performing backwards stepwise selection test.
    '''

    return ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'concavity_mean', 'concave points_mean', 'symmetry_mean']


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    gradient_model = GradientBoostingClassifier(
        n_estimators=10, learning_rate=0.1, max_depth=10, random_state=42)
    gradient_model.fit(X_train, (y_train))
    y_pred = gradient_model.predict(test_data)

    return y_pred
