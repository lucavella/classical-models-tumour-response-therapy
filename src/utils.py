import math
from enum import Enum
import numpy as np



class Trend(Enum):
    UP = 1
    DOWN = 2
    FLUCTUATE = 3


#converts the time to weeks
#e.g if the value is 227, this will return 33 since it happend in the 33th week of treatment
def correct_time_vector(time, convertToWeek = True):
    if convertToWeek:
        #days are converted to in which week they occured
        time = [math.ceil(i/7) for i in time]
        #if the value is negative, make it 0.1, otherwise keep the correct week
        time = [0.1 if i<=0 else i for i in time]
    else:
        time = [0.1 if i<=0 else i for i in time]
    return time

#if a value is not numeric, convert it to 'with_value'
def clean_nonnumeric(vector, with_value):
    #predicate to check if string is an integer
    def is_number(string):
        try:
            return not math.isnan(float(string))
        except ValueError:
            return False

    return [
        i if is_number(i) else with_value
        for i in vector
    ]


# detect if the trend of LD data is going, up, down or fluctuates
# based on paper section "Patient categorization according to RECIST and trajectory type"
# input: data point vector
# output: trend enum
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