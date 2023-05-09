import pandas as pd
import multiprocessing as mp
import itertools as it
import fitting as fit



# fit model to patient data
def fit_patient(model, patient):
    return fit.fitted_model(
        model, 
        patient['TreatmentDay'], 
        patient['TumorVolumeNorm']
    )


# fit model using "first_n" data points
# if "first_n" is set to None, all data points are used
def fit_study_params(study, model, first_n=None):
    def first_n_fitted_params(p):
        if first_n:
            p = p.head(first_n)

        params = fit.fitted_params(model, p['TreatmentDay'], p['TumorVolumeNorm'])
        if params is not None:
            return pd.Series(params)

    result = study.groupby(['StudyNr', 'Arm', 'PatientID']) \
                  .apply(first_n_fitted_params) \
                  .reset_index()
    
    study_nr = int(result.iloc[0]['StudyNr'])
    model_name = model.__name__.lower()
    name = f'study{study_nr}_{model_name}'

    return (result, name)
    

def save_study_params(studies, models, first_n=None, prefix='', max_workers=None):
    def store(result):
        df, name = result

        if first_n:
            amount = f'atleast{first_n}'
        else:
            amount = 'all'

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
                args=(study, model, first_n),
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

    #nwarnings.filterwarnings('ignore')

    # read all the studies as dataframes
    studies = [
        pd.read_excel(f'./data/study{i}.xlsx')
        for i in range(1, 6)
    ]
    study_names = ["FIR", "POPULAR", "BIRCH", "OAK", "IMvigor 210"]
    model_list = [
        models.Exponential,
        models.LogisticVerhulst,
        models.Gompertz,
        models.GeneralGompertz,
        models.ClassicBertalanffy,
        models.GeneralBertalanffy
    ]

    processed_studies = pre.preprocess(studies)

    save_study_params(
        processed_studies, 
        model_list, 
        prefix='./data/params/experiment1_initial', 
        max_workers=mp.cpu_count()
    )
