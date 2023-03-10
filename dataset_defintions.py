"""
File Overview
--------------
Contains constants relating the dataset that will be commonly referred to in the codebase
"""

# Column Related defintions
BEHAVIOR = 'BEHAVIOR'
ACCELERATION_X = 'ACCX'
ACCELERATION_Y = 'ACCY'
ACCELERATION_Z = 'ACCZ'
NEW_INDEX = 'INPUT_INDEX'
COLUMN_NAMES = [ BEHAVIOR,
            ACCELERATION_X, 
            ACCELERATION_Y, 
            ACCELERATION_Z ]

# Behavior that has no x,y,z
NO_VIDEO = 'n'

# Configuration
WINDOW_SIZE = 25
CLASSES_OF_INTEREST = "hlmst" #Testing larger windows with no strikes 17MAY22
#CLASSES_OF_INTEREST = "ilsw"

#Testing larger windows with no strikes 17MAY22
CLASS_INDICES = {  
    'h' : 0,
    #'i' : 1,
    'l' : 1,
    'm':  2,
    's' : 3,
    't' : 4
}
# CLASS_INDICES = {
#     'i' : 0,
#     'l' : 1,
#     's' : 2,
#     'w' : 3,
# }
# Post bucketizing of classes of interest
CLASSES_OF_INTEREST_LIST = ['l','s','t']# Original but testing larger windows and ignoring strike 17MAY22
#CLASSES_OF_INTEREST_LIST = ['l','s','i','w']

STRIKES = ['h', 'm']

# Models
RANDOM_FOREST = "Random_Forest"
SVM = "SVM"
NAIVE_BAYES = "Naive_Bayes"
ADABOOST_CLASSIFIER = "AdaBoost_Classifier"
ENSEMBLE_SVM = "Ensemble_SVM"

MODEL_NAMES = {
    "rf"  : RANDOM_FOREST,
    "svm" : SVM,
    "nb"  : NAIVE_BAYES,
    "abc" : ADABOOST_CLASSIFIER,
    "ens" : ENSEMBLE_SVM
}

# Types of Sampling
NO_SAMPLING = "No_Sampling"
SMOTE = "SMOTE"
ADASYN = "ADASYN"
RANDOM_OVERSAMPLE = "Random_Over_Sampler"
RANDOM_UNDERSAMPLE = "Random_Under_Sampler"


SAMPLING_TECHNIQUE_NAMES = {
    "o": RANDOM_OVERSAMPLE,
    "u": RANDOM_UNDERSAMPLE,
    "s" : SMOTE,
    "a" : ADASYN,
    'ns' : NO_SAMPLING,
}
