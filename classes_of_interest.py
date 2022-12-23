"""
File Overview
--------------
Returns behavior class letters present in the Data
"""

import pandas as pd
import numpy as np


def class_identifier(df, c_o_i):
    """
    desc
        Checks that the classes of intereest are present in the data and 
        returns a list of classes present in the data
    params
        df - dataframe of cleaned input data
        c_o_i - classes of interest argument
    return
        behavior_dict - dictionary of behaviors and their number translation, alphabetical, 1-n
        coi_list - list of present classes in input data
    """
    if c_o_i == False:
        behavior_dict = dict(zip(sorted(list(df.BEHAVIOR.unique().sum())), range(1, len(list(df.BEHAVIOR.unique().sum()))+1)))
        coi_list = list(behavior_dict.keys())
    else:
        blist = sorted(list(df.BEHAVIOR.unique().sum()))
        coi_list = ['other-classes'] + [bclass for bclass in c_o_i]
        behavior_dict = {x: 0 for x in blist}
        count = 0
        for bclass in c_o_i:
            count +=1
            behavior_dict[bclass] = count
        diff = list(set(c_o_i)-set(blist))
        if len(diff) > 0:
            missingclasses = ','.join(str(c) for c in diff)
            print("Classes " + missingclasses + " not found in input data.")

    return behavior_dict, coi_list
