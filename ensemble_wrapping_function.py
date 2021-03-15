# Importing models

import sys
import bagging_decision_trees
import decision_trees
import gradient_boosting
import knn
import logistic_regression
import random_forest
import svm_linear
import svm_rbf

# Importing libraries for transforming data
from datetime import datetime
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import clean_split_data
import seaborn as sns

sns.set()

def main():
    """
    Main function to read terminal input
    """
    system_argumets = sys.argv
    try:
        if system_argumets[1] == "-averaging":
            val = input("Enter your data filename as filename.csv: ")
            print("Starting the training and predicting process, this may take a few moments.")
            average_ensemble(val)
    except Exception as e:
        print(str(e))
        print("Either illegal arguments or no arguments were given by the User. Please read Readme file")


# Averaging ensemble function
def average_ensemble(csv_name):
    '''
    Takes user input data and returns excel with diagnosis of each patient. User must input csv file name including
    .csv file type as a string.
    '''
    data = pd.read_csv(str(csv_name))
    predictions_df = pd.DataFrame(columns=['Sample ID', 'Diagnosis'])
    list_of_models = [bagging_decision_trees, decision_trees, gradient_boosting,
                      knn, logistic_regression, random_forest, svm_linear, svm_rbf]
    list_of_votes = np.zeros(len(data)) 
    list_of_boolean_values = pd.DataFrame()
    
    #Generating Votes From Each Model
    for model in list_of_models:
        X = data[model.feature_names()]
        list_of_votes += (model.predict(X))
        
        boolean_list = pd.DataFrame(model.predict(X)).T
        list_of_boolean_values = list_of_boolean_values.append(boolean_list)
    index = 0

    #Determining Ensemble Vote
    for case in list_of_votes:
        average_vote = case / len(list_of_models)
        last_row = len(predictions_df)
        if average_vote < 0.5:
            predictions_df.loc[last_row] = [data['id'][index], 'B']
        else:
            predictions_df.loc[last_row] = [data['id'][index], 'M']
        index += 1
    predictions_df.set_index('Sample ID', inplace=True)
    
    #creating a PDF of Generated Data 
    len_predictions = len(list_of_boolean_values.T)
    fig_length = 8
    fig_height = 4*len_predictions
    fig,axes = plt.subplots(len_predictions,1, figsize =(fig_length,fig_height))
    
    x_ticks_ = [0,1]
    x_tick_labels = ['Benign','Malignant']
    
    with PdfPages("Sample_Prediction_Overview_"+str(datetime.now())+".pdf") as pdf:
        for i in range(len(list_of_boolean_values.T)):
            current_patient = list(list_of_boolean_values[i])
            sample_id = predictions_df.index[i]

            dictionary_of_patient_data = {'Prediction Density': current_patient}
            dataframe_of_patient_data = pd.DataFrame(dictionary_of_patient_data)

            mu = dataframe_of_patient_data['Prediction Density'].mean()
            median = np.median(dataframe_of_patient_data)
            sigma = dataframe_of_patient_data['Prediction Density'].std()

            stats_textstr = '\n'.join((
                r'Mean = %.2f' % (mu, ),
                r'Median = %.2f' % (median, ),
                r'STD = %.2f' % (sigma, )))

            if sigma == 0:
                dataframe_of_patient_data.plot(kind='hist',ax=axes[i])
                figure_title = "Sample ID: "+str(sample_id)+""
                axes[i].set_title(figure_title, fontsize=16, fontweight = 'bold')
                axes[i].set_xlabel('Diagnosis', fontsize=14)
                axes[i].set_xticks(x_ticks_)
                axes[i].set_xticklabels(x_tick_labels)
                axes[i].set_ylabel('Frequency', fontsize=14)
                axes[i].legend(bbox_to_anchor=(1.05, 1),loc='upper left', fontsize='large')
                axes[i].text(1.05, 0.5, stats_textstr, transform=axes[i].transAxes, fontsize=14,
                verticalalignment='center')
            else:
                dataframe_of_patient_data.plot(kind='density',ax=axes[i])
                figure_title = "Sample ID: "+str(sample_id)+""
                axes[i].set_title(figure_title, fontsize=16, fontweight = 'bold')
                axes[i].set_xlabel('Diagnosis', fontsize=14)
                axes[i].set_xticks(x_ticks_)
                axes[i].set_xticklabels(x_tick_labels)
                axes[i].set_ylabel('Density', fontsize=14)
                axes[i].legend(bbox_to_anchor=(1.05, 1),loc='upper left', fontsize='large')
                axes[i].text(1.05, 0.5, stats_textstr, transform=axes[i].transAxes, fontsize=14,
                verticalalignment='center')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close('all')
        print(predictions_df)
        print(".PDF and .CSV files saved under: Sample_Prediction_Overview_"+str(datetime.now()))
    
    return predictions_df, predictions_df.to_csv("Sample_Prediction_Overview_"+str(datetime.now())+".csv", index=True)

if __name__ == "__main__":
    main()


