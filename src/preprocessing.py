import pandas as pd
import itertools as it
import utils



# pairwise check if patient ID is reused across studies
# input: list of studies as dataframes
# output: False if the patient ID's are disjoint, True otherwise
def check_patient_overlap(studies):
    for study1, study2 in it.combinations(studies, 2):
        # pairwise inner join to check if empty
        if study1.join(study2, on='PatientID', rsuffix='_2', how='inner').size > 0:
            return True
    return False


# perform preprocessing as described in paper
# input: list of studies as dataframes
# output: combined preprocessed dataframe
def preprocess(studies):
    for study in studies:
        # set nonnumeric values to 0
        study['TargetLesionLongDiam_mm'] = utils.clean_nonnumeric(study['TargetLesionLongDiam_mm'], with_value=0)
        study['TargetLesionLongDiam_mm'].astype('float')

        # calculate tumor volume using formula
        study['TumorVolume_mm3'] = study['TargetLesionLongDiam_mm'].apply(lambda ld: ld ** 3 * 0.5)

    # merge all dataframes
    studies = pd.concat(studies, ignore_index=True)

    # normalize tumor volume to range of [0,1]
    min_tv = studies['TumorVolume_mm3'].min()
    max_tv = studies['TumorVolume_mm3'].max()
    studies['TumorVolumeNorm'] = studies['TumorVolume_mm3'].apply(lambda tv: (tv - min_tv) / (max_tv - min_tv))

    # extract study and arm nr
    studies['StudyNr'] = studies['StudyArm'].apply(lambda saTxt: int(saTxt[6]))
    studies['Arm'] = studies['StudyArm'].apply(lambda saTxt: int(saTxt[-1]))
    studies.drop('StudyArm', axis=1, inplace=True)

    return studies


# get all records of patients with 'i' or more data points
# input: joint dataframe of all studies, like the output of 'preprocess'
# output: dataframe with data points of patients with 'i' or more data points
def get_at_least(studies, i):
    return studies.groupby('PatientID') \
                  .filter(lambda group: group['PatientID'].count() >= i)


if __name__ == '__main__':
    study1 = pd.read_excel('../Original Paper/studies/Study1.xlsx')
    study2 = pd.read_excel('../Original Paper/studies/Study2.xlsx')
    study3 = pd.read_excel('../Original Paper/studies/Study3.xlsx')
    study4 = pd.read_excel('../Original Paper/studies/Study4.xlsx')
    study5 = pd.read_excel('../Original Paper/studies/Study5.xlsx')
    studies = [study1, study2, study3, study4, study5]
    
    print('Patient ID overlap across studies:', check_patient_overlap(studies))
    preprocessed = preprocess(studies)
    print(preprocessed)
    print(get_at_least(preprocessed, 6))