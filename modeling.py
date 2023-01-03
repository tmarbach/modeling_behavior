"""
File Overview
--------------
Entry point for running a selected model and sampling technique against the acceleration dataset
"""

import argparse

# Data Read and Initial Clean-up
from clean_acceleration_data import clean_dataset 

# Constants used to refer to the dataset post clean-up
from dataset_defintions import *

#Behavior Classes of Interest or Present in Data
from classes_of_interest import class_identifier

# Data Tranformations and Model Preparation
from prepare_acceleration_data import reduce_sample_dimension, stratified_shuffle_split, apply_sampling
from window_maker import leaping_window
from transformations import transform_accel_xyz

# Models
from model import naive_bayes, svm, random_forest, adaboost_classifier, ensemble_SVM

# Reporting
from reporting_acceleration_data import *



def retrieve_arguments():
    parser = argparse.ArgumentParser(
            prog='model_select', 
            description="Select a ML model to apply to acceleration data",
            epilog="") 
    parser.add_argument(
            "-i",
            "--raw_accel_csv",
            help = "input the path to the csv file or directory of accelerometer data that requires cleaning",
            type=str)
    parser.add_argument(
            "-m",
            "--model",
            help = "Choose a ML model: rf -- Random Forest, svm -- SVM, nb -- Naive Bayes, abc -- AdaBoost, ens -- Ensemble SVM",
            default="rf", 
            type=str)
    parser.add_argument(
            "-c",
            "--classes_of_interest",
            help="Define the classes of interest",
            default=False, 
            type=str)
    parser.add_argument(
            "-o",
            "--oversample",
            help = "Flag to oversample the minority classes: o -- oversample, s -- SMOTE, or a -- ADASYN ",
            default='ns', 
            type=str)
    parser.add_argument( 
            "-w",
            "--window_size",
            help="Number of rows to include in each data point (25 rows per second)",
            default=False, 
            type=int)
    parser.add_argument(
            "-d",
            "--data_output_file",
            help="Directs the data output to a filename of your choice",
            default=False)
    parser.add_argument(
            "-s",
            "--save_model",
            help="Once trained, saves the selected model as a .sav file for use on unlabeled data",
            default=False)
    return parser.parse_args()



def run_model(model_selection, X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag = False):
    """
    desc
        Runs the selected model with the train and test data
    params
        model_selection - selected model
        X_train         - samples to train on
        X_test          - samples to test on
        y_train         - ground truth of training data
        y_test          - ground truth of test data
        save_flag       - (optional) saves the trained model and outputs to a .sav file
    return
        report          - includes accuracy, percision per-class, 
                          recall per-class, macro average and weigted average
        y_pred          - Predictions made by the model on the test data
        classes         - list of class labels used by the model
    """
    if model_selection == 'svm':
        report, y_pred, classes = svm(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag)
    elif model_selection == 'nb':
        report, y_pred, classes= naive_bayes(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag)
    elif model_selection == 'abc':
        report, y_pred, classes = adaboost_classifier(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag)
    elif model_selection == 'ens':
        report, y_pred, classes = ensemble_SVM(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag)
    else: # Default to random forest model
        report, y_pred, classes = random_forest(X_train, X_test, y_train, y_test, CLASSES_OF_INTEREST_LIST, save_flag)

    return report, y_pred, classes


def prepare_train_test_pipeline(windowed_X_data, y_data, sample_flag=None):
    """
    desc
        Pipeline created to prepare windowed data, apply sampling techniques and
        split into test and train. 
    params
        windowed_X_data - X data that has had a windowing technique applied
        y_data          - ground truth
        sample_flag     - sampling technique to be applied
    return
        x_train_resampled - re-portioned X data from the sampling technique selected
        x_test - X data used to test the model after training
        y_train_resampled - re-portioned y data from the sampling technique selected (matches X data)
        y_test - Y data used to test the model after training
    """
    x_data_reduced = reduce_sample_dimension(windowed_X_data)
    x_train, x_test, y_train, y_test = stratified_shuffle_split(x_data_reduced, y_data)

    # reduce dimension sampler only does the sampling technique on the training data.
    x_train_resampled, y_train_resampled = apply_sampling(x_train, y_train, sample_flag)

    return x_train_resampled, x_test, y_train_resampled, y_test


def main(args):
    # Clean dataset based on input data, classes of interest, and sampling arguments
    df = clean_dataset(args.raw_accel_csv)
    classdict, presentclasses = class_identifier(df, args.classes_of_interest)
    windows, classes_found = leaping_window(df, args.window_size, args.classes_of_interest)
    Xdata, ydata, accurate_behavior_list = transform_accel_xyz(windows, classdict) #classdict was previously CLASS_INDICES
    X_train, X_test, y_train, y_test = prepare_train_test_pipeline(Xdata, ydata, args.oversample)
    
    # Run Model based on argument selection, and with present classes
    report, y_pred, classes = run_model(args.model, X_train, X_test, y_train, y_test, accurate_behavior_list, args.save_model)
    configuration_name = create_configuration_name(args.model, args.oversample, args.window_size, args.classes_of_interest)
    confusion_matrix = create_confusion_matrix(y_test, y_pred, classes)
    save_results(confusion_matrix, report, accurate_behavior_list, configuration_name)
    create_key_file(accurate_behavior_list, args.data_output_file)


if __name__ == "__main__":
    args = retrieve_arguments()
    main(args)

