import numpy as np
from dataset_defintions import *
#from sklearn.feature_selection import SelectFromModel (to be used to perform variables of importance)


def transform_accel_xyz(windows, class_dict):
    """
    desc:
        Converts list of single class dataframes to two arrays
        Xdata (all raw/transformed datapoints) & ydata (class label).
        transformations included per axis :
            mean, std, min, max, kurtosis, skew, corr (xy, yz, xz)
        Outputs a list of behavior letters in the order of numeric labels 
        assigned by the model. Use to translate numbers back to behavior letters.
    params:
        windows -- list of dataframes of all one class
        class_dict -- dictionary of classes {class letter: numeric value}
    return:
        Xdata -- arrays of x,y,z data of each window stacked together
        ydata -- integer class labels for each window
        accurate_behavior_list -- list of behaviors in order of numeric labels given by the model
    """
    positions = [ACCELERATION_X, ACCELERATION_Y, ACCELERATION_Z]

    Xdata, ydata, accurate_behavior_list = [], [], []
    for window in windows:
        data = np.empty((0,3), int)
        # Take the mean and standard deviation of each X, Y, and Z
        data = np.append(data, np.float32([window[positions].mean(axis = 0)]), 0)
        data = np.append(data, np.float32([window[positions].std(axis = 0)]),  0)

        Xdata.append(data)
        behavior = window[BEHAVIOR].iloc[0]

        # Bucketing the behavior to be categorized as strike
        if behavior in STRIKES:
            # t designates a general strike behavior
            behavior = 't'

        ydata.append(class_dict[behavior])
    
    values, counts = np.unique(ydata, return_counts=True)
    # Removes classes that are not present in the windows
    filtered_dict = {k:v for (k,v) in class_dict.items() if v != 0}
    # organizes the behaviors to match the numeric labels output by ConfMatrix
    for (k, v) in sorted(filtered_dict.items(), key=lambda item: item[1]):
        # This is removes the h and m labels from the data entirely and 
        # outputs appropriate labels for the ConfMatrix
        if v in values:
            accurate_behavior_list.append(k)
    
    return np.stack(Xdata), np.asarray(ydata), accurate_behavior_list