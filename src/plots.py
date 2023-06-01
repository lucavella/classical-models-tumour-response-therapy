import pandas as pd
import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

import models
import utils
import preprocessing as pre


# plot the change in LD from treatment start for given study
# corresponds to figure 1C
# input: names of studies, list of studies, amount of patients per study
def plot_change_trend(studies, amount=15, recist=True):
    # use RECIST 1.1 categories
    if recist:
        detect_f = utils.detect_recist
        ts = utils.Recist
        trend_name = 'RECIST'
    # categories proposed by the authors
    else:
        detect_f = utils.detect_trend
        ts = utils.Trend
        trend_name = 'trend'

    fig, axs = plt.subplots(1, len(studies), figsize=(18, 4))
    
    for (name, study), ax in zip(studies.items(), axs):
        # patients need >= 2 data points
        study = utils.get_at_least(
            utils.filter_treatment_started(study), 
            2
        )
        
        # take up to "amount" patients from study
        for patient in study['PatientID'].unique()[:amount]:
            # get LD and treatment week since treatment started for patient
            patient_data = study.loc[study['PatientID'] == patient]
            time = utils.convert_to_weeks(patient_data['TreatmentDay'])
            ld_data = np.array(patient_data['TargetLesionLongDiam_mm'])

            # get trend for color and LD deltas
            trend = detect_f(ld_data)
            ld_delta = ld_data - ld_data[0] # change in LD from first measurement
            time_delta = time - time[0] # start with time is 0
            

            # create subplot
            ax.axhline(y=0, linewidth=1, color='k', zorder=-1)
            ax.plot(
                time_delta,
                ld_delta,
                marker='o',
                markeredgewidth=1,
                linewidth=1,
                color=trend.color()
            )

            ax.set_title(name, fontsize=18, wrap=True)
            ax.set_xlabel('Time (weeks)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

    axs[0].set_ylabel('Change in LD from baseline (mm)', fontsize=16)
    axs[0].legend(
        [Line2D([0], [0], color=trend.color(), lw=4) for trend in ts], 
        [trend.name for trend in ts], 
        fontsize=16
    )
    fig.tight_layout()
    fig.savefig(f'../imgs/1C_{trend_name}.svg', format='svg', dpi=600)


# plot the proportions of trends for a study
# corresponds to figure 1D, but barchart instead of a nested pie chart for readability
def plot_proportion_trend(studies, recist=True):
    # use RECIST 1.1 categories
    if recist:
        detect_f = utils.detect_recist
        ts = utils.Recist
        trend_name = 'RECIST'
    # categories proposed by the authors
    else:
        detect_f = utils.detect_trend
        ts = utils.Trend
        trend_name = 'trend'


    fig, axs = plt.subplots(1, len(studies), figsize=(18, 4))

    for (name, study), ax in zip(studies.items(), axs):
        # count amount of patients per trend and arm
        trend_counts = study.groupby(['Arm', 'PatientID']) \
                            .apply(lambda p: detect_f(p['TargetLesionLongDiam_mm'])) \
                            .rename('Trend').reset_index() \
                            .groupby('Arm')['Trend'] \
                            .value_counts()

        # set trend categories that do not appear to 0
        arms = trend_counts.index.get_level_values('Arm').unique()
        trend_counts = trend_counts.reindex(
            pd.MultiIndex.from_product([arms, list(ts)]), 
            fill_value=0
        )

        # plot for each trend
        width = 0.2
        n_trends = len(ts)
        offsets = np.linspace( # calculate bar offsets
            width / 2 - n_trends / 10, # min offset
            - width / 2 + n_trends / 10, # max offset
            num=n_trends
        )
        for trend, offset in zip(ts, offsets):
            # get count for each arm and plot
            trend_count = trend_counts.loc[pd.IndexSlice[:, trend]]
            ax.bar(
                np.array(arms) + offset,
                trend_count, 
                width=width, 
                label=trend.name, 
                color=trend.color()
            )
        
        # create plot
        ax.set_xticks(arms)
        ax.set_xlabel('Study arms', fontsize=16)
        ax.set_title(name, fontsize=18, wrap=True)
        ax.tick_params(axis='both', which='major', labelsize=14)

    axs[0].set_ylabel('Number of occurences', fontsize=16)
    axs[0].legend(loc='upper left', fontsize=16)
    fig.tight_layout()
    fig.savefig(f'../imgs/1D_{trend_name}.svg', format='svg', dpi=600)

  
# plot the proportions of correct trends predictions based on up to "up_to_nth" data point per study and arm
# corresponds to figure 1E
def plot_correct_predictions(studies, up_to_nth=4, recist=True):
    # use Recist 1.1 categories
    if recist:
        detect_f = utils.detect_recist
        trend = 'RECIST'
    # categories proposed by the authors
    else:
        detect_f = utils.detect_trend
        trend = 'trend'
    
    nth_points = np.arange(2, up_to_nth + 2) # baseline does not count
    merged_studies = pd.concat(studies.values(), ignore_index=True)
    merged_studies = utils.get_at_least(
        utils.filter_treatment_started(merged_studies), 
        2
    )

    # get trend of each patient
    trends = merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                           .apply(lambda p: detect_f(p['TargetLesionLongDiam_mm'])) \
                           .rename('Trend')


    # get proportions of correct trends from 1 extra to "up_to" extra data points, per arm
    data = [
        merged_studies.groupby(['StudyNr', 'Arm', 'PatientID']) \
                      .apply(lambda p: \
                          # compare trend of first i with final trend
                          detect_f(p.head(n)['TargetLesionLongDiam_mm']) \
                          == trends.loc[*p.name]
                      ) \
                      .rename('CorrectTrend').reset_index() \
                      .groupby(['StudyNr', 'Arm'])['CorrectTrend'] \
                      .aggregate('mean')
        for n in nth_points
    ]

    data_med = [np.median(x) for x in data]

    # create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    bplts = ax.boxplot(data, positions=nth_points - 1, notch=True, patch_artist=True)
    ax.set_xlabel('nth data point used to predict', fontsize=16)
    ax.set_ylabel(f'Proportion of correct {trend} predictions', fontsize=16)
    # ax.set_title(f'Proportion of correct {trend} predictions using only data point 1 up to only data point {up_to_nth}', fontsize=20, wrap=True)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # color fill
    cmap = cm.ScalarMappable(cmap='plasma')
    # print(cmap)
    for patch, med in zip(bplts['boxes'], data_med):
        #print(color)
        color = cmap.to_rgba(med)
        patch.set_facecolor(color)

    fig.tight_layout()
    fig.savefig(f'../imgs/1E_{trend}.svg', format='svg', dpi=600)

    print(f'MEDIAN {trend}: {data_med}')


# plot actual vs predicted normalized tumor volume values
# corresponds to figure 2C
def plot_actual_fitted(studies, models, dirname, log_scale=True, part=None):
    def get_params(params, p):
        return np.array(params.loc[params['PatientID'] == p].iloc[0, 3:])
    
    def predict(p):
        p_params = get_params(params, p.name)
        if np.isnan(p_params).any():
            pred = [math.nan] * len(p['TreatmentDay'])
        else:
            pred = model.predict(p['TreatmentDay'], *p_params)
        return pd.Series(pred)

    fig, axs = plt.subplots(len(studies), len(models), figsize=(12, 12))

    for i, ((name, study), ax_row) in enumerate(zip(studies.items(), axs), start=1):
        study = utils.get_at_least(
            utils.filter_treatment_started(study), 
            3
        )
        
        for model, ax in zip(models, ax_row):
            params = pd.read_csv(f'{dirname}/study{i}_{model.__name__.lower()}.csv')

            # predict model function per patient
            predicted = study.groupby('PatientID') \
                             .apply(predict) \
                             .rename('PredictedTumorVolumeNorm').reset_index() \
                             .join(study, rsuffix='_') \
                             .dropna()

            # create subplot
            ax.scatter(predicted['TumorVolumeNorm'], predicted['PredictedTumorVolumeNorm'], s=2)
            ax.axline((0, 0), slope=1, linestyle=':', linewidth=1, color='k', zorder=10)
            ax.tick_params(axis='both', which='major', labelsize=14)
            if log_scale:
                ax.set_xscale('log')
                ax.set_yscale('log')
            # ax.set_xlabel('Actual normalized tumor volume', fontsize=12)
            # ax.set_ylabel('Predicted normalized tumor volume', fontsize=12)
        ax_row[0].set_ylabel(name, fontsize=18)

    for ax, model in zip(axs[0], models):
        ax.set_title(model.__name__, fontsize=18)

    fig.tight_layout()

    if part:
        filename = f'../imgs/2C_{part}.svg'
    else:
        filename = '../imgs/2C.svg'
    fig.savefig(filename, format='svg', dpi=600)


def plot_trend_pred_error(studies, models, dirname, experiment, error_metric='MAE', recist=True):
    def get_params(params, p):
        return np.array(params.loc[params['PatientID'] == p].iloc[0, 3:])
    
    def predict(p):
        p_params = get_params(params, p.name)
        if np.isnan(p_params).any():
            pred = [math.nan] * len(p['TreatmentDay'])
        else:
            pred = model.predict(p['TreatmentDay'], *p_params)
        return pd.Series(pred)
    
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
        study = utils.filter_treatment_started(study)
        if experiment == 1:
            study = utils.get_at_least(study, 3)
        elif experiment == 2:
            study = utils.get_at_least(study, 6)

        study_trends = study.groupby('PatientID') \
                            .apply(lambda p: detect_f(p['TumorVolumeNorm'])) \
                            .rename('Trend').reset_index() \
                            .merge(study, on='PatientID', how='left')

        for model in models:
            params = pd.read_csv(f'{dirname}/study{i}_{model.__name__.lower()}.csv')

            # predict model function per patient
            predicted = study_trends.groupby('PatientID') \
                                    .apply(predict) \
                                    .rename('PredictedTumorVolumeNorm').reset_index() \
                                    .join(study_trends, rsuffix='_') \
                                    .dropna()

            # get the prediction error per study and trend
            pred_error = predicted.groupby('Trend') \
                                  .apply(lambda t: error_f(model, t)) \
                                  .rename('Error') \
                                  .sort_index()
            
            pred_error.index = pred_error.index.map(lambda t: f'{name} {t}')

            study_results[model.__name__] = pred_error

        results.append(study_results)

    results = pd.concat(results)
    
    fig, ax = plt.subplots(figsize=(15, 15))

    ax = sns.heatmap(results, annot=True, annot_kws={'fontsize': 16}, ax=ax)
    # ax.set_title(f'Experiment {experiment} {error_metric} by final {trend_name}', fontsize=20, wrap=True)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    fig.axes[-1].tick_params(labelsize=16)
    fig.align_xlabels()
    fig.tight_layout()
    fig.savefig(f'../imgs/3_{error_metric}_exp{experiment}.svg', format='svg', dpi=600)
    


if __name__ == "__main__":
    rc('font', family='serif', serif=['Computer Modern'])
    rc('text', usetex=True)

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
        models.Logistic,
        models.GeneralLogistic,
        models.Gompertz,
        models.GeneralGompertz,
        models.ClassicBertalanffy,
        models.GeneralBertalanffy,
        models.DynCarryingCapacity
    ]

    processed_studies = {
        name: study
        for name, study in zip(study_names, pre.preprocess(studies))
    }

    plot_change_trend(processed_studies)
    plot_change_trend(processed_studies, recist=False)

    plot_proportion_trend(processed_studies)
    plot_proportion_trend(processed_studies, recist=False)

    plot_correct_predictions(processed_studies)
    plot_correct_predictions(processed_studies, recist=False)

    plot_actual_fitted(
        processed_studies, 
        models[:4], 
        './data/params/experiment1_odeint', 
        part=1
    )
    plot_actual_fitted(
        processed_studies, 
        models[4:], 
        './data/params/experiment1_odeint', 
        part=2
    )
    
    # plot_trend_pred_error(
    #     processed_studies, 
    #     models,
    #     experiment=1,
    #     dirname='data/params/experiment1_odeint',
    #     error_metric='MAE'
    # )
    
    # plot_trend_pred_error(
    #     processed_studies, 
    #     models,
    #     experiment=1,
    #     dirname='data/params/experiment1_odeint',
    #     error_metric='AIC'
    # )
    
    # plot_trend_pred_error(
    #     processed_studies, 
    #     models,
    #     experiment=2,
    #     dirname='data/params/experiment2_odeint',
    #     error_metric='MAE'
    # )
    
    # plot_trend_pred_error(
    #     processed_studies, 
    #     models,
    #     experiment=2,
    #     dirname='data/params/experiment2_odeint',
    #     error_metric='AIC'
    # )
    
    # plot_trend_pred_error(
    #     processed_studies, 
    #     models,
    #     experiment=2,
    #     dirname='data/params/experiment2_odeint',
    #     error_metric='R2'
    # )
    
    
