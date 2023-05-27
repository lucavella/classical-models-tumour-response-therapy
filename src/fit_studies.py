import pandas as pd
import multiprocessing as mp
import itertools as it
import fitting as fit
import utils



# fit model to patient data
def fit_patient(model, patient):
    return fit.fitted_model(
        model, 
        patient['TreatmentDay'], 
        patient['TumorVolumeNorm']
    )


# fit model without last "drop_last" data points
# if "drop_last" is set to None, all data points are used
def fit_study_params(study, model, experiment):
    def drop_last_fitted_params(p):
        if experiment == 2:
            # drop last 3 measurements if experiment 2
            p = p.head(len(p) - 3)

        params = fit.fitted_params(model, p['TreatmentDay'], p['TumorVolumeNorm'])
        if params is not None:
            return pd.Series(params)

    study = utils.filter_treatment_started(study)
    if experiment == 1:
        study = utils.get_at_least(study, 3)
    elif experiment == 2:
        study = utils.get_at_least(study, 6)

    result = study.groupby(['StudyNr', 'Arm', 'PatientID']) \
                  .apply(drop_last_fitted_params) \
                  .reset_index()
    
    study_nr = int(result.iloc[0]['StudyNr'])
    model_name = model.__name__.lower()
    name = f'study{study_nr}_{model_name}'

    return (result, name)
    

# fits data points without last "drop_last" of each patient for all combinations of studies and models and stores as csv
# runs parallelized for efficiency
def save_study_params(studies, models, experiment, prefix='', max_workers=None):
    def store(result):
        df, name = result

        if experiment == 1:
            amount = 'all'
        elif experiment == 2:
            amount = 'drop3'

        df.to_csv(
            f'{prefix}{name}_{amount}.csv'
        )

    if max_workers:
        workers = max_workers
    else:
        workers = len(studies) * len(models)

    s_studies = sorted(studies, key=len, reverse=True)
    
    with mp.Pool(processes=workers) as pool:
        results = [
            pool.apply_async(
                fit_study_params,
                args=(study, model, experiment),
                callback=store,
                error_callback=print
            )
            for study in s_studies
            for model in models
        ]

        for r in results:
            r.wait()
    


if __name__ == "__main__":
    import warnings
    import preprocessing as pre
    import models

    # warnings.filterwarnings('ignore')

    # read all the studies as dataframes
    studies = [
        pd.read_excel(f'./data/study{i}.xlsx')
        for i in range(1, 6)
    ]

    model_list = [
        models.Exponential,
        models.Logistic,
        models.GeneralLogistic,
        models.Gompertz,
        models.GeneralGompertz,
        models.ClassicBertalanffy,
        models.GeneralBertalanffy,
        models.DynamicCarryingCapacity
    ]

    processed_studies =  pre.preprocess(studies)

    # save_study_params(
    #     processed_studies, 
    #     model_list,
    #     experiment=1,
    #     prefix='./data/params/experiment1_ivp/', 
    #     max_workers=mp.cpu_count()
    # )

    save_study_params(
        processed_studies,
        model_list,
        experiment=2,
        prefix='./data/params/experiment2_ivp/', 
        max_workers=mp.cpu_count()
    )