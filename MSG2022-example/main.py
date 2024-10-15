# built-in
import os

# third-party
import numpy as np
import h5py
from sklearn import linear_model
import biosppy as bp

# SeFEF
from sefef import evaluation, labeling

# local 
from data import get_metadata, create_hdf5_dataset
from config import patient_id, data_folder_path, sampling_frequency, features2extract


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
    files = files[:1000]

    ## Segment files
    create_hdf5_dataset(files, dataset_filepath=os.path.join(preprocessed_data_path, f'dataset.h5'), sampling_frequency=sampling_frequency, features2extract=features2extract)
    
if __name__ == '__main__':
    main()