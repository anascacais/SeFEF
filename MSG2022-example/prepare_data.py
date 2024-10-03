# built-in
import os
from datetime import datetime, timezone

# third-party
import pandas as pd
import numpy as np
from sefef import evaluation 

# local 
from data import get_sz_onset_df
from config import patient_id, data_folder_path, sampling_frequency


def main(data_folder_path=data_folder_path, patient_id=patient_id, sampling_frequency=sampling_frequency):

    # Convert file metadata and sz onsets into expected format
    train_labels_all = pd.read_csv(os.path.join(data_folder_path, 'train_labels.csv'))
    train_labels_all[['id', 'session', 'utc-datetime']] = train_labels_all['filepath'].str.split('/', expand=True)
    train_labels_all['utc-datetime'] = train_labels_all['utc-datetime'].str.strip('.parquet')

    train_labels = train_labels_all.loc[train_labels_all['id'] == patient_id, :]

    ## Convert UTC datetime to Unix timestamp
    train_labels['utc-datetime'] = pd.to_datetime(train_labels['utc-datetime'], format='UTC-%Y_%m_%d-%H_%M_%S', utc=True)
    files_metadata = pd.DataFrame({
        'filepath': train_labels.reset_index()['filepath'], 
        'first_timestamp': train_labels.reset_index()['utc-datetime'].astype('int') // 10**9, # Convert to seconds
    }) 
    files_metadata['total_duration'] = [10*60] * len(files_metadata) # 10min in seconds

    ## Get seizure onsets
    sz_onset_df = get_sz_onset_df(train_labels.reset_index()[['utc-datetime', 'label']].to_numpy(), mpl=10) # mpl set to 10min to account for the pre-ictal labeling done in MSG2022
    sz_onsets = (sz_onset_df.index.astype('int') // 10**9).tolist()

    dataset = evaluation.Dataset(files_metadata, sz_onsets, sampling_frequency=sampling_frequency)
    
    tscv = evaluation.TimeSeriesCV()
    tscv.split(dataset, iteratively=False)
    # for train_start_ts, test_start_ts, test_end_ts in tscv.split(dataset):
    #     pass

    tscv.plot(dataset)




if __name__ == '__main__':
    main()