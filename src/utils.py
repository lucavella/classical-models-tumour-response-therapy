import math
from enum import Enum
import numpy as np
import itertools as it
import pandas as pd 
import fitting as fit
import warnings



# get all records of patients with 'i' or more data points
# params: study dataframe
# return: dataframe with data points of patients with 'i' or more data points
def get_at_least(study, i):
    return study.groupby('PatientID') \
                  .filter(lambda group: group['PatientID'].count() >= i)


# pairwise check if patient ID is reused across studies
# params: list of studies as dataframes
# return: False if the patient ID's are disjoint, True otherwise
def check_patient_overlap(studies):
    for study1, study2 in it.combinations(studies, 2):
        # pairwise inner join to check if empty
        if study1.join(study2, on='PatientID', rsuffix='_2', how='inner').size > 0:
            return True
    return False


# converts the time (days) to weeks
# e.g if the day 227 => week 32.43
# params: time vector in days
# return: time vector in weeks
def convert_to_weeks(time):
    return [i/7 for i in time]


# removes measurements before treatment started
# params: time and measurements
# return: time and measurements since treatment started
def filter_treatment_started(time, data):
    time = np.array(time)
    data = np.array(data)

    treatment_started = time >= 0
    return (
        time[treatment_started],
        data[treatment_started]
    )


# Trend enum
class Trend(Enum):
    UP = 1
    FLUCTUATE = 2
    DOWN = 3

    def color(self):
        if self == Trend.UP:
            return '#d73027'
        elif self == Trend.FLUCTUATE:
            return '#313695'
        elif self == Trend.DOWN:
            return '#1a9850'

    def __lt__(self,other):
        return self.value < other.value
    
class Trend_recist(Enum):
    CR = 1
    PR = 2
    PD = 3
    SD = 4

    def __lt__(self,other):
        return self.value < other.value
    
    
# class to print colors in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    

# detect if the trend of LD data is going, up, down or fluctuates
# based on paper section "Patient categorization according to RECIST and trajectory type"
# params: data point vector
# return: trend enum
def detect_trend(vector):
    # get difference vector
    v = np.array(vector)
    diff_v = v[1:] - v[:-1]

    # get sum of positive and negative differences
    pos = np.sum(
        np.clip(diff_v, a_min=0, a_max=None)
    )
    neg = - np.sum(
        np.clip(diff_v, a_min=None, a_max=0)
    )
    
    # UP if strictly positive or sum of positive to sum of negative rate is > 2
    if (neg == 0) or (pos / neg > 2):
        return Trend.UP
    # DOWN if strictly negative or sum of negative to sum of positive rate is > 2
    elif (pos == 0) or (neg / pos > 2):
        return Trend.DOWN
    # FLUCTUATE else
    else:
        return Trend.FLUCTUATE
    
def detect_trend_recist(vector):
    # get difference vector
    v = np.array(vector)
    #calculate difference between last diameter and first measured diameter
    difference = v[-1] - v[0]
    #CR: (Complete Response) dissapearing of all target lesions
    if v[-1] == 0:
        return Trend_recist.CR
    #PR: (Partial Response) at least 30% decrease in diameter
    elif difference < - 0.3  * v[0]:
        return Trend_recist.PR
    #PD: (Progressive Disease) at least 20% increase in diamater
    elif difference > 0.2 * v[0]:
        return Trend_recist.PD
    #SD: (Stable disease) none of the above apply
    else:
        return Trend_recist.SD
    
    
    
def split_on_trend(study):
    up = []
    down = []
    fluctuate = []
    
    for patient in study['PatientID'].unique():
        patient_data = study.loc[study['PatientID'] == patient]
        ld_data = np.array(patient_data['TargetLesionLongDiam_mm'])
        trend = detect_trend(ld_data)
        if trend == Trend.UP:
            up.append(patient_data)
        elif trend == Trend.DOWN:
            down.append(patient_data)
        elif trend == Trend.FLUCTUATE:
            fluctuate.append(patient_data)
   
    
    return pd.concat(up), pd.concat(down), pd.concat(fluctuate)

def split_on_trend_recist(study):
    complete_response = []
    partial_response = []
    progressive_disease = []
    stable_disease = []
    
    for patient in study['PatientID'].unique():
        patient_data = study.loc[study['PatientID'] == patient]
        ld_data = np.array(patient_data['TargetLesionLongDiam_mm'])
        trend = detect_trend_recist(ld_data)
        if trend == Trend_recist.CR:
            complete_response.append(patient_data)
        elif trend == Trend_recist.PR:
            partial_response.append(patient_data)
        elif trend == Trend_recist.PD:
            progressive_disease.append(patient_data)
        elif trend == Trend_recist.SD:
            stable_disease.append(patient_data)
    
    return pd.concat(complete_response), pd.concat(partial_response), pd.concat(progressive_disease), pd.concat(stable_disease)
