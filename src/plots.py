import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils
import preprocessing as pre
import fitting as fit



# plot the change in LD from baseline for given study
# corresponds to figure 1C
def plot_change_trend(name, study, amount=10):
    study = utils.get_at_least(study, 2) # patients need >= 2 data points
    fig, ax = plt.subplots()
    
    # take up to "amount" patients from study
    for patient in study['PatientID'].unique()[:amount]:
        # get LD and treatment week from dataframe for patient
        patient_data = study.loc[study['PatientID'] == patient]
        ld_data = np.array(patient_data['TargetLesionLongDiam_mm'])
        time = utils.convert_to_weeks(patient_data['TreatmentDay'])

        # get trend for color and LD deltas
        trend = utils.detect_trend(ld_data)
        ld_delta = ld_data - ld_data[0] # change in LD from first measurement
        time_delta = np.array(time) - time[0] # start with time is 0
        
        ax.plot(
            time_delta,
            ld_delta,
            marker='o',
            markeredgewidth=3,
            linewidth=2,
            color=trend.color()
        )

    # create plot labels, legend, ...
    plt.axhline(y=0, linestyle=':', color='black')
    plt.xlabel('Time (weeks)', fontsize=16)
    plt.ylabel('Change in LD from baseline (mm)', fontsize=16)
    plt.title(f'Change in LD and trend per patient for {name}', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(
        [Line2D([0], [0], color=trend.color(), lw=4) for trend in utils.Trend], 
        [trend.name for trend in utils.Trend], 
        fontsize=16
    )
    plt.show()


# plot the proportions of trends for a study
# corresponds to figure 1D, but barchart instead of a nested pie chart for readability
def plot_proportion_trend(name, study):
    # count amount of patients per trend and arm
    trend_counts = study.groupby(['Arm', 'PatientID']) \
                        .apply(lambda p: utils.detect_trend(p['TargetLesionLongDiam_mm'])) \
                        .rename('Trend').reset_index() \
                        .groupby('Arm')['Trend'] \
                        .value_counts()

    # set trend categories that do not appear to 0
    arms = np.array(trend_counts.index.get_level_values('Arm').unique())
    trend_counts = trend_counts.reindex(
        pd.MultiIndex.from_product([arms, list(utils.Trend)]), 
        fill_value=0
    )

    # plot for each trend
    width = 0.2
    n_trends = len(utils.Trend)
    offsets = np.linspace(width / 2 - n_trends / 10, - width / 2 + n_trends / 10, num=n_trends) # calculate bar offsets
    for trend, offset in zip(utils.Trend, offsets):
        # get count for each arm and plot
        trend_count = trend_counts.loc[pd.IndexSlice[:, trend]]
        plt.bar(
            arms + offset,
            trend_count, 
            width=width, 
            label=trend.name, 
            color=trend.color()
        )
    
    plt.xticks(arms)
    plt.xlabel("Study arms", fontsize=16)
    plt.ylabel("Number of occurences", fontsize=16)
    plt.title(f'Trend categories per Study Arm for {name}', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.show()

    
def fig_1E(studies):    
    first_datapoint_prediction = []
    second_datapoint_prediction = []
    third_datapoint_prediction = []
    fourth_datapoint_prediction = []
    
    for study in studies:
        study_arms = list(study['StudyArm'].unique())
        for arm in study_arms:
            counter = 0
            filteredData = study.loc[study['StudyArm'] == arm]
            patientIDs = list(filteredData['PatientID'].unique())
            amount_of_patients = len(list(filteredData['PatientID'].unique()))
            
            #true en falses bijhouden van de correcte voorspellingen
            first_point = []
            second_point = []
            third_point = []
            fourth_point = []
            X = [first_point,second_point,third_point,fourth_point]
            
            #patienten aflopen in de study arm
            while counter < amount_of_patients:
                key = patientIDs[counter]
                filteredDataPatient = study.loc[study['PatientID'] == key]
                
                #voor elk amount of datapoints de voorspelling doen
                for idx, point in enumerate(X):
                    if len(filteredDataPatient) >= idx + 1:
                        datapoints = list(filteredDataPatient['TargetLesionLongDiam_mm']) 
                        time = list(filteredDataPatient['TreatmentDay'])

                        time = utils.convert_to_weeks(time)
                        datapoints = utils.clean_nonnumeric(datapoints, with_value = 0) #convert the mi
                        datapoints = [x for _,x in sorted(zip(time,datapoints))]
                        time.sort()
                        trend = p.detect_trend(datapoints)
                        
                        #check for certain amount of datapoints
                        restricted_data_point = datapoints[0:idx + 1]
                        restricted_time = time[0:idx + 1]
                        restricted_time = utils.convert_to_weeks(restricted_time)
                        
                        restricted_data_point = utils.clean_nonnumeric(restricted_data_point, with_value = 0) #convert the mi
                        restricted_data_point = [x for _,x in sorted(zip(restricted_time,restricted_data_point))]
                        
                        restricted_time.sort()
                        restricted_trend = p.detect_trend(restricted_data_point)
                        
                        #voorspelling is hetzelfde = true
                        if trend ==  restricted_trend:
                            point.append(True)
                        #voorspelling is niet hetzelfde = false
                        else:
                            point.append(False)
                counter += 1
                
            #hier append van probabiliteiten in de lijsten
            first_datapoint_prediction.append(round(sum(first_point)/len(first_point) * 100, 2))
            second_datapoint_prediction.append(round(sum(second_point)/len(second_point) * 100, 2))
            third_datapoint_prediction.append(round(sum(third_point)/len(third_point) * 100, 2))
            fourth_datapoint_prediction.append(round(sum(fourth_point)/len(fourth_point) * 100, 2))
    
    data = [first_datapoint_prediction, second_datapoint_prediction, third_datapoint_prediction, fourth_datapoint_prediction]
    
    fig,ax = plt.subplots()
    
    # Creating plot
    bp = ax.boxplot(data, positions=[1,2,3,4])
    ax.set_ylabel('probability (%)')
    ax.set_xlabel('amount of datapoints used to predict')
    ax.set_title("probability of correct prediction with 1-4 datapoints")
    # show plot
    plt.show()

if __name__ == "__main__":
    # read all the studies as dataframes
    studies = [
        pd.read_excel(f'./data/study{i}.xlsx')
        for i in range(1, 6)
    ]
    study_names = ["FIR", "POPULAR", "BIRCH", "OAK", "IMvigor 210"]

    processed_studies = pre.preprocess(studies)
    for name, study in zip(study_names, processed_studies):
        plot_change_trend(name, study, amount=10)
        plot_proportion_trend(name, study)
    # fig_1D(study_name='BIRCH', study=study_3)
    # fig_1E(studies)
