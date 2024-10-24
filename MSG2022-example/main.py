# built-in
import os

# third-party
import numpy as np
import h5py
from sklearn import linear_model, preprocessing

# SeFEF
from sefef import evaluation, labeling, postprocessing, scoring

# local 
from data import get_metadata, create_hdf5_dataset
from config import patient_id, data_folder_path, sampling_frequency, features2extract, metrics2compute


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    preprocessed_data_path=f'data/preprocessed_data/{patient_id}'

    files_metadata, sz_onsets = get_metadata(data_folder_path, patient_id)

    # SeFEF - evaluation module 
    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=sampling_frequency)
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=False, plot=False)

    # Segmentation
    ## List all files for segmentation
    files = dataset.metadata.loc[np.min(tscv.split_ind_ts) : np.max(tscv.split_ind_ts)]['filepath']
    files = [os.path.join(data_folder_path, file) for file in files]


    # Segment files
    # if not os.path.exists(os.path.join(preprocessed_data_path, f'dataset.h5')):
    if not os.path.exists(preprocessed_data_path):
        os.makedirs(preprocessed_data_path)
    create_hdf5_dataset(files, dataset_filepath=os.path.join(preprocessed_data_path, f'dataset.h5'), sampling_frequency=sampling_frequency, features2extract=features2extract)
    
    # SeFEF - labeling module
    with h5py.File(os.path.join(preprocessed_data_path, f'dataset.h5'), 'r+') as h5dataset:
        if 'annotations' not in h5dataset.keys():
            labeling.add_annotations(h5dataset, sz_onsets_ts=dataset.metadata[dataset.metadata['sz_onset']==1].index.to_numpy())
        if 'sz_onsets' not in h5dataset.keys():
            labeling.add_sz_onsets(h5dataset, sz_onsets_ts=dataset.metadata[dataset.metadata['sz_onset']==1].index.to_numpy())


    # Operationalizing CV
    with h5py.File(os.path.join(preprocessed_data_path, f'dataset.h5'), 'r') as h5dataset:
        
        performance = {m:[] for m in metrics2compute}
        for ifold, (train_data, test_data) in enumerate(tscv.iterate(h5dataset)):
            print(f'\n---------------------\nStarting TSCV fold {ifold+1}/{tscv.n_folds}\n---------------------')
                                
            X_train, y_train, _ = train_data
            X_test, y_test, ts_test, sz_onsets_test = test_data

            # Apply scaling
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Logistic regression
            model = linear_model.LogisticRegression(class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)

            # SeFEF - postprocessing module
            y_pred = model.predict_proba(X_test)
            forecast = postprocessing.Forecast(y_pred[:, 1], ts_test)
            #forecast = postprocessing.Forecast(y_test, ts_test)
            forecasts, ts = forecast.postprocess(forecast_horizon=60*60, smooth_win=5*60, origin='clock-time')
            
            # SeFEF - scoring module
            scorer = scoring.Scorer(metrics2compute=metrics2compute, sz_onsets=sz_onsets_test, forecast_horizon=60*60, reference_method='naive')
            fold_performance = scorer.compute_metrics(forecasts, ts, draw_diagram=True, binning_method='equal_forecast_number')
            
            for metric in fold_performance.keys():
                performance[metric].append(fold_performance[metric])
        
        pass

if __name__ == '__main__':
    main()