"""
File Overview
--------------
Functionaility for composing and optionally saving metrics
"""

import pandas as pd
from dataset_defintions import * 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def create_configuration_name(selected_model, selected_sampling, window_size, coi):
    """
    desc
        create a name for the configuration being done
    params
        selected_model      - the model that will be run
        selected_sampling   - the sampling technique applied to the testing data
    return
        configuration_name  - a string indicting the model and sampling technique selected
    """
    model_name = MODEL_NAMES[selected_model]
    sampling_technique_name = SAMPLING_TECHNIQUE_NAMES[selected_sampling]
    if coi == False:
        configuration_name = str(model_name + '_' + sampling_technique_name + '_' + str(window_size) + '_all')
    configuration_name = str(model_name + '_' + sampling_technique_name + '_' + str(window_size) + '_' + coi)    
    

    return configuration_name


def create_confusion_matrix(y_test, y_pred, classes):
    """
    desc
        Creates and labels the confusion matrix
    params
        y_test  - the base truth
        y_pred  - the model's predictions
        classes - the classes the model was asked to predict
    returns
        Confusion matrix for the given data
    """
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    return cm



def create_key_file(classes, file_name):
    """
    desc
        writes a text file with the classes present in the run in the proper order
        this is needed for labeling wild data with the proper behavior letters
    params
        classes - string of letters that represent behaviors able to be predicted by the trained model
        file_name - name and location of the text file containing the key
    """
    with open(file_name, 'w') as f:
        f.write(str(classes))


def save_results(confusion_matrix, report, classes_present, configuration_name="output"): 
    """
    desc
        Save all outputs of the model performance on test data by saving the 
        confusion matrix and evaluation reports to disk.
    params
        confusion_matrix    - confusion matrix of the model on the testing data
        report              - report of the model performance on testing data
        configuration_name  - (optional) name used when writing the figure and report
    """
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes_present)
    cm_display.plot()
    cm_display.ax_.set_title(configuration_name)
    #plt.savefig(str("./figures/" + configuration_name + '.png')) line from original, could change to check for folder
    plt.savefig(str(configuration_name + '.png'))

    report = pd.DataFrame(report).transpose()
    
    # The index is lost when being written to an excel sheet
    # Add index back in as a column at the beginning of the dataframe
    report.insert(0, "Index", report.index)

    #result_path = str('./results/' + configuration_name + '.xlsx')
    result_path = str(configuration_name + '.xlsx') 
    with pd.ExcelWriter(result_path, engine='xlsxwriter') as writer:
        report.to_excel(writer, index=False)