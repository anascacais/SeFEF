# built-in
import os

# third-party
import numpy as np
import h5py
from sklearn import linear_model, preprocessing

# SeFEF
from sefef import evaluation, labeling, postprocessing, scoring

# local
from data import get_metadata, create_timeseries_dataset
from config import patient_id, data_folder_path, sampling_frequency, features2extract, metrics2compute


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    preprocessed_data_path = f'./data_files/preprocessed_data/{patient_id}'

    files_metadata, sz_onsets = get_metadata(data_folder_path, patient_id)

    # SeFEF - evaluation module
    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=sampling_frequency)
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=False, plot=True)

    # Segmentation
    # List all files for segmentation
    files = dataset.metadata.loc[np.min(tscv.split_ind_ts): np.max(tscv.split_ind_ts)]['filepath']
    files = [os.path.join(data_folder_path, file) for file in files]

    # Segment files
    if not os.path.exists(os.path.join(preprocessed_data_path, f'dataset.h5')):
        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)
        create_timeseries_dataset(files, dataset_filepath=os.path.join(preprocessed_data_path, f'dataset.h5'),
                            sampling_frequency=sampling_frequency)


if __name__ == '__main__':
    main()