# Project Title

modeling_behavior is a python script that trains scikit-learn shallow machine learning
 algorithms to predict behaviors from labeled triaxial accelerometry data in csv or xlsx format. 

## Description

Using open source software scikit-learn machine learning and sampling strategy libraries
 in conjunction with pandas and numpy, we created modeling_behavior: a script that trains
 MLAs using labeled triaxial accelerometry data with optional sampling strategies to mitigate 
the effects of behavioral class imbalance. Behavior classes to be found can be specified, temporal 
window sizes must be selected, and the summary statistics are fixed to mean, min/max, standard 
deviation, and overall dynamic body acceleration (ODBA). Sampling strategies can be used to 
counteract data imbalanace and boost minority class identification. Five options of MLA are
avialable to choose from. 

## Getting Started

### Dependencies

Python 3.7
scikit-learn 1.0
numpy 1.19
pandas 1.1
pickle 0.7
imbalanced-learn 0.7
matplotlib 3.3

### Installing

Clone this repo using the command:

git clone https://github.com/tmarbach/modeling_behavior.git

(works best in a conda environment)

### Executing program

modeling.py -i input_data_file/ -m model -o u -classes_of_interest behavior-code -window_size integer_value -d abc_slwt_25undersample_mso_transf

usage: model_select [-h] [-i RAW_ACCEL_CSV] [-m MODEL]
 [-c CLASSES_OF_INTEREST] [-o OVERSAMPLE] [-w WINDOW_SIZE]
 [-d DATA_OUTPUT_FILE] [-s SAVE_MODEL]

Select a ML model to apply to acceleration data
optional arguments:
-h, --help
show this help message and exit
-i RAW_ACCEL_CSV, --raw_accel_csv RAW_ACCEL_CSV
input the path to the csv file or directory of
accelerometer data that requires cleaning
-m MODEL, --model MODEL
Choose a ML model: rf -- Random Forest, svm -- SVM,
	 nb -- Naive Bayes, abc -- AdaBoost,
	 ens -- Ensemble SVM
-c CLASSES_OF_INTEREST, --classes_of_interest CLASSES_OF_INTEREST
Define the classes of interest
-o OVERSAMPLE, --oversample OVERSAMPLE
Flag to oversample the minority classes: o -- oversample,
	 u -- undersample, s -- SMOTE, or a -- ADASYN
-w WINDOW_SIZE, --window_size WINDOW_SIZE
Number of rows to include in each data point (25 rows per second)
-d DATA_OUTPUT_FILE, --data_output_file DATA_OUTPUT_FILE
Directs the data output to a filename of your choice
-s SAVE_MODEL, --save_model SAVE_MODEL
Once trained, saves the selected model as a .sav file                               
for use on unlabeled data

## Help

Do not use the save_model flag without knowing 
that the model is useful.
In order for the models to run, each behavior must 
have at enough data points for both training and testing.


## Authors

Tyler Marbach
tymarbach@gmail.com

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

