import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import Utils as utils
from matplotlib.lines import Line2D
import math
import extra_functions as ef
import re 

#read all the studies in a dataframe
study_1 = pd.read_excel('./Original Paper/studies/Study1.xlsx') #FIR
study_2 = pd.read_excel('./Original Paper/studies/Study2.xlsx') #POPLAR
study_3 = pd.read_excel('./Original Paper/studies/Study3.xlsx') #BIRCH
study_4 = pd.read_excel('./Original Paper/studies/Study4.xlsx') #OAK
study_5 = pd.read_excel('./Original Paper/studies/Study5.xlsx') #IMvigor 210

#list of the dataframes
studies = [study_1, study_2, study_3, study_4, study_5]

def calculate_patients():
    patients_with_three_or_more_datapoints = 0
    patients_with_six_or_more_datapoints = 0

    #loop over all dataframes
    for study in studies:
        for patient in study['Patient_Anonmyized'].value_counts():
            if patient >= 6:
                patients_with_three_or_more_datapoints += 1
                patients_with_six_or_more_datapoints += 1
            elif patient >= 3:
                patients_with_three_or_more_datapoints +=1
            else:
                print(patient)
                
    print(f'patients with three or more datapoints: {patients_with_three_or_more_datapoints}')
    print(f'patients with six or more datapoints: {patients_with_six_or_more_datapoints}')


#plot fig 1C study1
#This plot is to show that all categories are present
def fig_1C(studyname, study, amount_of_patients = 10):
    patientID = list(study['Patient_Anonmyized'].unique()) #get all unique patients in this study
    counter = 0 #used to iterate over the patients
    fig, ax = plt.subplots()

    while counter <= amount_of_patients:
        key = patientID[counter]
        filteredData = study.loc[study['Patient_Anonmyized'] == key] #check data per patient

        if len(filteredData) >= 2: #patients needs to have more than 2 datapoints
            datapoints = list(filteredData['TargetLesionLongDiam_mm']) 
            time = list(filteredData['Treatment_Day'])

            time = ef.correct_time_vector(time, convertToWeek = True) #convert the days to weeks
            datapoints = ef.remove_string_from_numeric_vector(datapoints, valueToReplace = 0) #convert the missing values to zero

            datapoints = [x for _,x in sorted(zip(time,datapoints))]
            time.sort()
            trend = ef.detect_trend_of_data(datapoints)
            print(trend)   
            new_dim = [datapoints[0]] * len(datapoints)
            change = [a_i - b_i for a_i, b_i in zip(datapoints, new_dim)]

            if trend ==  'Up':
                ax.plot(time, change, marker='o', markeredgewidth = 3, linewidth = 2, color='#d73027')
            elif trend == 'Down':
                ax.plot(time, change, marker='o', markeredgewidth = 3, linewidth = 2, color='#1a9850')
            elif trend == "Fluctuate":
                ax.plot(time, change, marker='o', markeredgewidth = 3, linewidth = 2, color='#313695')
        counter += 1

    #create the plot
    plt.axhline(0, linestyle = '--', color='black')
    plt.xlabel('Time (weeks)', fontsize = 16)
    plt.ylabel('Change in  LD From Baseline(mm)', fontsize = 16)
    plt.title(studyname, fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    custom_lines = [Line2D([0], [0], color='#d73027', lw=4),
                    Line2D([0], [0], color='#1a9850', lw=4),
                    Line2D([0], [0], color='#313695', lw=4)]

    ax.legend(custom_lines, ['Up', 'Down', 'Fluctuate'], fontsize = 12)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()
    
    
def fig_1D(study_name ,study):
    study_arms = list(study['Study_Arm'].unique()) #get all subgroups of patients
    
    up_list = []
    down_list = []
    fluctuate_list = []
    
    for arm in study_arms:
        counter = 0
        up_counter = 0
        down_counter = 0
        fluctuate_counter = 0
        filteredData = study.loc[study['Study_Arm'] == arm]
        patientID = list(filteredData['Patient_Anonmyized'].unique())
        amount_of_patients = len(list(filteredData['Patient_Anonmyized'].unique()))
        
        
        while counter < amount_of_patients:
            key = patientID[counter]
            filteredDataPatient = study.loc[study['Patient_Anonmyized'] == key]
            if len(filteredDataPatient) >= 2: #patients needs to have more than 2 datapoints
                datapoints = list(filteredDataPatient['TargetLesionLongDiam_mm']) 
                time = list(filteredDataPatient['Treatment_Day'])
                
                time = ef.correct_time_vector(time, convertToWeek = True) #convert the days to weeks
                datapoints = ef.remove_string_from_numeric_vector(datapoints, valueToReplace = 0) #convert the mi
                datapoints = [x for _,x in sorted(zip(time,datapoints))]
                time.sort()
                trend = ef.detect_trend_of_data(datapoints)
                 
                if trend ==  'Up':
                    up_counter +=1
                elif trend == 'Down':
                    down_counter +=1
                elif trend == "Fluctuate":
                    fluctuate_counter +=1
            counter += 1
        up_list.append(up_counter)
        down_list.append(down_counter)
        fluctuate_list.append(fluctuate_counter)
        
    X_axis = np.arange(len(study_arms))
    
    study_arms = [study_arm[8:] for study_arm in study_arms]
    #X = [study.replace(f'{study_name}_', '') for study in X]
    
    plt.bar(X_axis - 0.2, up_list, 0.2, label = 'Up', color="red")
    plt.bar(X_axis, down_list, 0.2, label = 'Down', color="green")
    plt.bar(X_axis + 0.2, fluctuate_list, 0.2, label = 'Fluctuate', color="blue")
    
    plt.xticks(X_axis, study_arms)
    plt.xlabel("Study Arms")
    plt.ylabel("Number of occurences")
    plt.title(f'Categories per Study Arm for {study_name}')
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    #calculate_patients()
    #fig_1C(studyname="FIR", study=study_1, amount_of_patients = 10)
    #fig_1C(studyname="POPLAR", study=study_2, amount_of_patients = 10)
    #fig_1C(studyname="BIRCH", study=study_3, amount_of_patients = 100)
    #fig_1C(studyname="OAK", study=study_4, amount_of_patients = 10)
    #fig_1C(studyname="IMvigor 210", study=study_5, amount_of_patients = 10)
    fig_1D(study_name='BIRCH', study=study_3)
        
    



