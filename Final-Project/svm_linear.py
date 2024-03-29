# Pandas library for the pandas dataframes
import pandas as pd


# Import cleaning and splitting functions
from clean_split_data import clean_data
from clean_split_data import split_data

# Import Scikit-Learn library for the classification models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Import plotting libraries
import matplotlib

# Set larger fontsize for all plots
matplotlib.rcParams.update({'font.size': 20})

# ### Data
data = pd.read_csv('data/data.csv')
data = clean_data(data)
X_train, X_test, y_train, y_test = split_data(data)

# ### Classifier
clf = SVC(kernel='linear', C=9)
clf.fit(X_train, y_train)


# ### Optimized SVM_linear Classifier
def feature_names():
    '''
    Returns array of input features of
    best performing backwards stepwise selection test.'''

    return ['radius_mean',
            'texture_mean',
            'area_mean',
            'compactness_mean',
            'concavity_mean',
            'concave points_mean',
            'symmetry_mean',
            'fractal_dimension_mean']


def predict(test_data):
    '''
    Takes test data and uses classifier to predict boolean output.
    '''
    X = data[feature_names()]
    y = data.diagnosis
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = SVC(kernel='linear', C=9)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(test_data)

    return y_pred
