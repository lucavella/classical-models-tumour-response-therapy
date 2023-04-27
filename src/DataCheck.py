import pandas as pd

study_1 = pd.read_excel('./Original Paper/studies/Study1.xlsx')
study_2 = pd.read_excel('./Original Paper/studies/Study2.xlsx')
study_3 = pd.read_excel('./Original Paper/studies/Study3.xlsx')
study_4 = pd.read_excel('./Original Paper/studies/Study4.xlsx')
study_5 = pd.read_excel('./Original Paper/studies/Study5.xlsx')
#print(study_1)

#print(study_1['Patient_Anonmyized'].value_counts())
#print(study_2['Patient_Anonmyized'].value_counts())
#print(study_3['Patient_Anonmyized'].value_counts())
#print(study_4['Patient_Anonmyized'].value_counts())
#print(study_5['Patient_Anonmyized'].value_counts())

studies = [study_1, study_2, study_3, study_4, study_5]

three_or_more = 0
six_or_more = 0


for study in studies:
    for patient in study['Patient_Anonmyized'].value_counts():
        if patient > 5:
            three_or_more += 1
            six_or_more += 1
        elif patient > 2:
            three_or_more +=1
        else:
            print(patient)
            
print(f'three_or_more: {three_or_more}')
print(f'six_or_more: {six_or_more}')


#plot patient per study
item = 0
patientID = list(study_1['Patient_Anonmyized'].unique())
print(patientID)
key = patientID[item] #eerste patient pakken

filteredData = study_1.loc[study_1['Patient_Anonmyized'] == key]
print(filteredData)