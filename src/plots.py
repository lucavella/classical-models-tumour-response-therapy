import pandas as pd
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

import models
import utils
import preprocessing as pre


# plot the change in LD from treatment start for given study
# corresponds to figure 1C
# input: names of studies, list of studies, amount of patients per study
def plot_change_trend(studies, amount=10):
    fig, axs = plt.subplots(1, len(studies), figsize=(25, 5))
    
    for (name, study), ax in zip(studies.items(), axs):
        study = utils.get_at_least(study, 2) # patients need >= 2 data points
        
        # take up to "amount" patients from study
        for patient in study['PatientID'].unique()[:amount]:
            # get LD and treatment week since treatment started for patient
            patient_data = study.loc[study['PatientID'] == patient]
            time, ld_data  = utils.filter_treatment_started(
                utils.convert_to_weeks(patient_data['TreatmentDay']),
                patient_data['TargetLesionLongDiam_mm']
            )

            # get trend for color and LD deltas
            trend = utils.detect_trend(ld_data)
            ld_delta = ld_data - ld_data[0] # change in LD from first measurement
            time_delta = time - time[0] # start with time is 0
            

            # create subplot
            ax.plot(
                time_delta,
                ld_delta,
                marker='o',
                markeredgewidth=3,
                linewidth=2,
                color=trend.color()
            )

            ax.set_title(name, fontsize=20)
            ax.axhline(y=0, linestyle=':', linewidth=1, color='gray')
            ax.set_xlabel('Time (weeks)', fontsize=16)
            ax.set_ylabel('Change in LD from baseline (mm)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)

    # plt.legend(
    #     [Line2D([0], [0], color=trend.color(), lw=4) for trend in utils.Trend], 
    #     [trend.name for trend in utils.Trend], 
    #     fontsize=12
    # )

    fig.tight_layout()
    fig.savefig('../imgs/1C.svg', format='svg', dpi=1200)


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
def plot_correct_predictions(studies, up_to=5, recist=True):
    # use Recist 1.1 categories
    if recist:
        detect_f = utils.detect_recist
    # categories proposed by the authors
    else:
        detect_f = utils.detect_trend
    
    amount_points = range(2, up_to + 1) # always at least 2 points
    merged_studies = pd.concat(studies.values(), ignore_index=True)

    # get trend of each patient
    trends = merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                           .apply(lambda p: detect_f(p['TargetLesionLongDiam_mm'])) \
                           .rename('Trend')


    # get proportions of correct trends from 1 extra to "up_to" extra data points, per arm
    data = [
        merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                      .apply(lambda p: \
                          # compare trend of first i with final trend
                          detect_f(p.head(i)['TargetLesionLongDiam_mm']) \
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
def plot_actual_fitted(studies, models, dirname, log_scale=False):
    def get_params(params, p):
        return np.array(params.loc[params['PatientID'] == p].iloc[0, 4:])

    fig, axs = plt.subplots(len(studies), len(models), figsize=(25, 25))

    for i, ((name, study), ax_row) in enumerate(zip(studies.items(), axs), start=1):
        for model, ax in zip(models, ax_row):
            params = pd.read_csv(f'{dirname}/study{i}_{model.__name__.lower()}_all.csv')

            # predict model function per patient
            predicted = study.groupby('PatientID') \
                             .apply(lambda p: pd.Series(
                                model.predict(p['TreatmentDay'], *get_params(params, p.name))
                             ))

            # create subplot
            ax.scatter(study['TumorVolumeNorm'], predicted)
            ax.axline((0, 0), slope=1, linestyle=':', color='black')
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            ax.set_xlabel('Actual normalized tumor volume', fontsize=10)
            ax.set_ylabel('Predicted normalized tumor volume', fontsize=10)
            ax.set_title(f'Study: {name}, Model: {model.__name__}', fontsize=12)

    fig.tight_layout()
    fig.savefig('../imgs/2C.svg', format='svg', dpi=1200)


def plot_trend_pred_error(studies, models, dirname, error_metric='MAE', recist=True, normalize=True):
    def get_params(params, p):
        return np.array(params.loc[params['PatientID'] == p].iloc[0, 4:])
    
    def error_f(model, t):
        if error_metric == 'MAE':
            return mean_absolute_error(
                t['TumorVolumeNorm'], 
                t['PredictedTumorVolumeNorm']
            )
        elif error_metric == 'AIC':
            return utils.akaike_information_criterion(
                model.params * len(t['PatientID'].unique()),
                t['TumorVolumeNorm'], 
                t['PredictedTumorVolumeNorm']
            )
        elif error_metric == 'R2':
            return np.mean(
                t.groupby('PatientID') \
                 .apply(lambda p: r2_score(
                     p['TumorVolumeNorm'], 
                     p['PredictedTumorVolumeNorm']
                 ))
            )
    
    # use RECIST 1.1 categories
    if recist:
        detect_f = utils.detect_recist
        trend_name = 'RECIST'
    # categories proposed by the authors
    else:
        detect_f = utils.detect_trend
        trend_name = 'trend'

    model_names = list(map(lambda m: m.__name__, models))
    results = []

    for i, (name, study) in enumerate(studies.items(), start=1):
        study_results = pd.DataFrame(columns=model_names)

        # detect trend per patient
        study = utils.get_at_least(study, 6)
        study_trends = study.groupby('PatientID') \
                            .apply(lambda p:
                                f"{name} {detect_f(p['TumorVolumeNorm']).name}"
                            ) \
                            .rename('StudyTrend').reset_index() \
                            .merge(study, on='PatientID', how='left')

        for model in models:
            params = pd.read_csv(f'{dirname}/study{i}_{model.__name__.lower()}_all.csv')

            # predict model function per patient
            predicted = study_trends.groupby('PatientID') \
                                    .apply(lambda p: pd.Series(
                                        model.predict(p['TreatmentDay'], *get_params(params, p.name))
                                    )) \
                                    .rename('PredictedTumorVolumeNorm').reset_index() \
                                    .join(study_trends, rsuffix='_') \
                                    .dropna()

            # get the prediction error per study and trend
            pred_error = predicted.groupby('StudyTrend') \
                                  .apply(lambda t: error_f(model, t)) \
                                  .rename('Error') \
                                  .sort_index()

            study_results[model.__name__] = pred_error

        results.append(study_results)

    results = pd.concat(results)
    
    ax = sns.heatmap(results, annot=True, annot_kws={'fontsize': 16})
    ax.set_title(f'{error_metric} values categorized by final {trend_name}', fontsize=20)
    ax.tick_params(labelsize=16)

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='center')
    plt.show()
    


if __name__ == "__main__":
    # disable warning in terminal
    warnings.filterwarnings("ignore")
    # read all the studies as dataframes
    study_names = ['FIR', 'POPLAR', 'BIRCH', 'OAK', 'IMVIGOR210']
    studies = [
        pd.read_excel(f'./data/study{i}.xlsx')
        for i, _ in enumerate(study_names, start=1)
    ]
        
    models = [
        models.Exponential,
        models.LogisticVerhulst,
        models.Gompertz,
        models.GeneralGompertz,
        models.ClassicBertalanffy,
        models.GeneralBertalanffy
    ]

    processed_studies = {
        name: study
        for name, study in zip(study_names, pre.preprocess(studies))
    }

    # plot_correct_predictions(processed_studies, recist=True)

    # plot_change_trend(processed_studies)

    # for name, study in processed_studies.items():
    #     plot_proportion_trend(name, study)

    # plot_correct_predictions(processed_studies)

    # plot_actual_fitted(processed_studies, models, './data/params/experiment1_initial')
    
    plot_trend_pred_error(processed_studies, models, 'data/params/experiment1_initial', error_metric='R2')
    
    
