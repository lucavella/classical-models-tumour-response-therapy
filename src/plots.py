import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import models
import utils
import preprocessing as pre
import fit_studies as fit
from sklearn.metrics import mean_absolute_error
import warnings
from sklearn import preprocessing



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
    arms = trend_counts.index.get_level_values('Arm').unique()
    trend_counts = trend_counts.reindex(
        pd.MultiIndex.from_product([arms, list(utils.Trend)]), 
        fill_value=0
    )

    # plot for each trend
    width = 0.2
    n_trends = len(utils.Trend)
    offsets = np.linspace( # calculate bar offsets
        width / 2 - n_trends / 10, # min offset
         - width / 2 + n_trends / 10, # max offset
         num=n_trends
    )
    for trend, offset in zip(utils.Trend, offsets):
        # get count for each arm and plot
        trend_count = trend_counts.loc[pd.IndexSlice[:, trend]]
        plt.bar(
            np.array(arms) + offset,
            trend_count, 
            width=width, 
            label=trend.name, 
            color=trend.color()
        )
    
    # create plot
    plt.xticks(arms)
    plt.xlabel('Study arms', fontsize=16)
    plt.ylabel('Number of occurences', fontsize=16)
    plt.title(f'Trend categories per Study Arm for {name}', fontsize=24)
    plt.legend(fontsize=16)
    plt.show()

  
# plot the proportions of correct trends predictions based on 2 to "up_to" data points per study and arm
# corresponds to figure 1E
def plot_correct_predictions(studies, up_to=5):
    amount_points = range(2, up_to + 1) # always at least 2 points
    merged_studies = pd.concat(studies, ignore_index=True)

    # get trend of each patient
    trends = merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                           .apply(lambda p: utils.detect_trend(p['TargetLesionLongDiam_mm'])) \
                           .rename('Trend')


    # get proportions of correct trends from 1 extra to "up_to" extra data points, per arm
    data = [
        merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                      .apply(lambda p: \
                          # compare trend of first i with final trend
                          utils.detect_trend(p.head(i)['TargetLesionLongDiam_mm']) \
                          == trends.loc[*p.name]
                      ) \
                      .rename('CorrectTrend').reset_index() \
                      .groupby(['StudyNr', 'Arm'])['CorrectTrend'] \
                      .aggregate('mean')
        for i in amount_points
    ]

    # create plot
    plt.boxplot(data, positions=amount_points)
    plt.xlabel('Amount of first data points used to predict', fontsize=16)
    plt.ylabel('Proportion of correct predictions', fontsize=16)
    plt.title(f'Proportion of correct trend prediction with 2 up to {up_to} data points', fontsize=24)
    plt.show()


# plot actual vs predicted normalized tumor volume values
# corresponds to figure 2C
def plot_actual_fitted(study_names, studies, models, log_scale=False):
    fig, axs = plt.subplots(len(studies), len(models))

    for i, (study, name) in enumerate(zip(studies, study_names)):
        for j, model in enumerate(models):
            # fit and predict model function per patient
            predicted = study.groupby('PatientID') \
                             .apply(lambda p: pd.Series(
                                fits.fit_patient(model, p)(p['TreatmentDay'])
                             ))

            # create subplot
            axs[i, j].scatter(study['TumorVolumeNorm'], predicted)
            axs[i, j].axline((0, 0), slope=1, linestyle=':', color='black')
            if log_scale:
                axs[i, j].set_xscale('log')
                axs[i, j].set_yscale('log')
            axs[i, j].set_xlabel('Actual normalized tumor volume', fontsize=16)
            axs[i, j].set_ylabel('Predicted normalized tumor volume', fontsize=16)
            axs[i, j].set_title(f'Study: {name}, Model: {model.__name__}', fontsize=16)

    plt.title(f'Actual vs predicted normalized tumor volumes', fontsize=24)
    plt.show()


def get_mae_and_aic(study_names, studies, models):
    #column names of the output file
    df = pd.DataFrame(columns=['study_trend','model','MAE','AIC'])
    dictionary = dict()
    
    #split the studies in their recist groups (up/down/fluctuate)
    for i, study in enumerate(studies):
        study = utils.get_at_least(study, 6)
        up_name = study_names[i]+"_up"
        down_name = study_names[i]+"_down"
        fluctuate_name = study_names[i]+"_fluctuate"
        
        up, down, fluctuate = utils.split_on_trend(study)
        
        dictionary[up_name] = up
        dictionary[down_name] = down
        dictionary[fluctuate_name] = fluctuate
    
    #go over every dataframe in the dictionary    
    for entry in dictionary:
        for model in models:
            #predict with the model
            predicted = dictionary[entry].groupby('PatientID') \
                                         .apply(lambda p: pd.Series(
                                            fits.fit_patient(model, p)(p['TreatmentDay'])
                                         ))
            
            #check if the predictions have nan 
            if predicted.hasnans:
                for i, v in predicted.items():
                    if math.isnan(v):
                        #drop patients where prediction was NAN
                        dictionary[entry] = dictionary[entry][dictionary[entry]['PatientID'] != i[0]]
            
            #drop nan values            
            predicted = predicted.dropna()
            
            #convert predicted series to list to calculate absolute error (needs to be same datatype)
            frame_predicted = predicted.to_frame()
            frame_predicted = frame_predicted[0].to_list()
            tumorVolumeNorm_list = dictionary[entry]['TumorVolumeNorm'].to_list()

            Sum = 0
            n = len(frame_predicted)
            for i in range(n):
                Sum += abs(frame_predicted[i] - tumorVolumeNorm_list[i])
                
            absError = Sum/n
            SE = np.square(absError)
            temp_sum = np.sum(SE)
            #calculate aic
            aic = (2 * utils.model_parameters[model.__name__]) - (2 * np.log(temp_sum))   
            #calculate mae
            mae = mean_absolute_error(dictionary[entry]['TumorVolumeNorm'], predicted)
            new_row = {'study_trend':entry, 'model': model.__name__, 'MAE': mae, 'AIC':aic}
            df.loc[len(df)] = new_row
    df.to_csv("./src/data/output_MAE_AIC.csv")
    
    
 #function to create the heatmap of the MAE and AIC
def create_heatmap(file_path, normalize = False, value = "MAE"):
    data = pd.read_csv(file_path)
    if normalize:
        data[value] /= data.groupby("study_trend")[value].transform(sum)
        pivot = data.pivot(index='study_trend',columns='model',values=value)
        ax = sns.heatmap(pivot, annot= True)
    else: 
        pivot = data.pivot(index='study_trend',columns='model',values=value)
        ax = sns.heatmap(pivot, annot= True)
    plt.show()
    


if __name__ == "__main__":
    #disable warning in terminal
    warnings.filterwarnings("ignore")
    # read all the studies as dataframes
    studies = [
        pd.read_excel(f'./src/data/study{i}.xlsx')
        for i in range(1, 6)
    ]
    study_names = ["FIR", "POPULAR", "BIRCH", "OAK", "IMvigor 210"]
    models = [
        models.Exponential,
        models.LogisticVerhulst,
        models.Gompertz,
        models.GeneralGompertz,
        models.ClassicBertalanffy,
        models.GeneralBertalanffy
    ]

    processed_studies = pre.preprocess(studies)

    # for name, study in zip(study_names, processed_studies):
    #     plot_change_trend(name, study, amount=10)
    #     plot_proportion_trend(name, study)

    # plot_correct_predictions(processed_studies)

    #plot_actual_fitted(study_names, processed_studies, models)
    #heatmaps(study_names=study_names, studies=studies, models=models)
    create_heatmap(file_path="./src/data/output_MAE_AIC.csv", normalize=True, value="AIC")
    
    
