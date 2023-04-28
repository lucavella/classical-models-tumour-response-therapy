import math

#converts the time to weeks
#e.g if the value is 227, this will return 33 since it happend in the 33th week of treatment
def Correct_Time_Vector(time, convertToWeek = True):
    if convertToWeek:
        #days are converted to in which week they occured
        time = [math.ceil(i/7) for i in time]
        #if the value is negative, make it 0.1, otherwise keep the correct week
        time = [0.1 if i<=0 else i for i in time]
    else:
        time = [0.1 if i<=0 else i for i in time]
    return time

#predicate to check if string is an integer
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

#if a value is not numeric, convert it to 'valueToReplace'
def Remove_String_From_Numeric_Vector(vector, valueToReplace):
    vector = [valueToReplace if not is_number(str(i)) else i for i in vector]
    vector = [valueToReplace if  str(i) == 'nan' else i for i in vector]
    return vector

#detect if the trend of data is going, up, down or fluctuates
#based on what is explained in paper section "Patient categorization according to RECIST and trajectory type"  
def Detect_Trend_Of_Data(vector):
    diff = [] #difference of each LD measurement 
    for d in range(len(vector)-1):
        diff.append(vector[d + 1] - vector[d])  #difference of each LD measurement at timepoint t + 1 to its previous measurement at time point t for each patient
        
    s_pos = 0
    s_neg = 0
    
    for x in diff:
        if x > 0:
            s_pos = s_pos + x
        elif x < 0:
            s_neg = s_neg + x
            
    #if the second measurement was bigger each time, this means RECIST will be up or ratio >= 2         
    if all(i >= 0 for i in diff) or (diff[0] > 0 and s_pos/abs(s_neg) >= 2):
        trend = 'Up'
    #if the second measurement was smaller each time, this means RECIST will be down or ratio >= 2 
    elif all(i <= 0 for i in diff) or (diff[0] < 0 and abs(s_neg)/s_pos >= 2):
        trend = 'Down'
    #if not one of the above, the category is fluctuate    
    else:
        trend = 'Fluctuate'
    return trend







