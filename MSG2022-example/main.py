# local 
from prepare_data import prepare_data
from config import patient_id, data_folder_path, sampling_frequency


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    dataset, tscv = prepare_data(data_folder_path, patient_id, sampling_frequency)
    
    # Segmentation and feature extraction
    for train_start_ts, test_start_ts, test_end_ts in tscv.split(dataset):
        print(train_start_ts, test_start_ts, test_end_ts)


if __name__ == '__main__':
    main()