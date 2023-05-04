import math
from enum import Enum
import numpy as np



# converts the time (days) to weeks
# e.g if the day 227 => week 33
# input: time vector in days
# output: time vector in weeks
def convert_to_weeks(time):
    return [math.ceil(i/7) for i in time]


# if a value in vector not numeric, replace it to "with_value"
# input: vector
# output: numeric vector
def clean_nonnumeric(vector, with_value=0):
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


# Trend enum
class Trend(Enum):
    UP = 1
    DOWN = 2
    FLUCTUATE = 3