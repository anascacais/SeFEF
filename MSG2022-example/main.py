# built-in
import os

# SeFEF
from sefef import evaluation

# local 
from data import get_metadata, create_dataset
from config import patient_id, data_folder_path, sampling_frequency


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    preprocessed_data_path=f'data/preprocessed_data/{patient_id}'

    files_metadata, sz_onsets = get_metadata(data_folder_path, patient_id)

    # Evaluation module 
    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=sampling_frequency)
    
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=True)

    # Segmentation and feature extraction
    for ifold, (train_start_ts, test_start_ts, test_end_ts) in enumerate(tscv.split(dataset)):

        train_files = [os.path.join(data_folder_path, file) for file in dataset.metadata.loc[train_start_ts : test_start_ts]['filepath'].tolist()]
        test_files = [os.path.join(data_folder_path, file) for file in dataset.metadata.loc[test_start_ts : test_end_ts]['filepath'].tolist()]

        if not os.path.exists(preprocessed_data_path):
            os.makedirs(preprocessed_data_path)

        create_dataset(train_files, test_files, dataset_filepath=f'TSCV_{ifold+1}.h5', sampling_frequency=sampling_frequency)

     


if __name__ == '__main__':
    main()