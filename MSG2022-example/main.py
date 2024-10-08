# built-in
import os

# third-party
import numpy as np
import h5py
from sklearn import linear_model

# SeFEF
from sefef import evaluation, labeling

# local 
from data import get_metadata, create_hdf5_dataset
from config import patient_id, data_folder_path, sampling_frequency


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    preprocessed_data_path=f'data/preprocessed_data/{patient_id}'

    files_metadata, sz_onsets = get_metadata(data_folder_path, patient_id)

    # Evaluation module 
    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=sampling_frequency)
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=False, plot=False)

    # Segmentation
    ## List all files for segmentation
    files = dataset.metadata.loc[np.min(tscv.split_ind_ts) : np.max(tscv.split_ind_ts)]['filepath']
    files = [os.path.join(data_folder_path, file) for file in files]

    ## Segment files
    if not os.path.exists(os.path.join(preprocessed_data_path, f'dataset.h5')):
        create_hdf5_dataset(files, dataset_filepath=os.path.join(preprocessed_data_path, f'dataset.h5'), sampling_frequency=sampling_frequency)
    
    # Labeling module
    with h5py.File(os.path.join(preprocessed_data_path, f'dataset.h5'), 'r+') as h5dataset:
        if 'annotations' not in h5dataset.keys():
            labeling.add_annotations(h5dataset, sz_onsets_ts=dataset.metadata[dataset.metadata['sz_onset']==1].index.to_numpy())


    # Operationalizing CV
    with h5py.File(os.path.join(preprocessed_data_path, f'dataset.h5'), 'r+') as h5dataset:
        
        for ifold, (train_data, test_data) in enumerate(tscv.iterate(h5dataset)):
            print(f'\n---------------------\nStarting TSCV fold {ifold+1}/{tscv.n_folds}\n---------------------')
                                
            X_train, y_train, _ = train_data
            X_test, y_test, _ = test_data
            
            logr = linear_model.LogisticRegression()
            logr.fit(X_train, y_train)

            y_pred = logr.predict(X_test)
            pass

if __name__ == '__main__':
    main()