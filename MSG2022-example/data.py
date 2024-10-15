# built-in
import os
import datetime
import time

# third-party
import h5py
import numpy as np
import pandas as pd
import scipy

# local 
from features import extract_features, extract_features_bp


def get_sz_onset_df(labels_array, mpl=10):
    """Returns a dataframe with the seizure onset datetime from the labels of pre-ictal and inter-ictal data files.

    Parameters
    ----------
    labels_array: numpy array
        numpy array with dimension (points in sample, samples, 2),
            where the first dimension is the time series within each sample (with resolution=time_resolution) and
            the third dimension corresponds to the datetime of the start of that point (dtype='int64') as well as the label (dtype='int64')
    mpl: int
        minimum prediction latency in minutes

    Returns
    -------
    sz_onset_df: pandas datafram
        dataframe with the seizure onset datetimes
    """

    _, last_preictal_np_indx = get_indx_first_last_preictal_samples(
        labels_array)
    last_preictal_np_indx += 1

    sz_onset_df = pd.DataFrame(
        columns=["utc-datetime", "sz-onset"])
    preictal = np.take(labels_array, last_preictal_np_indx, axis=0)

    for ind in preictal:
        sz_onset_df.loc[len(sz_onset_df)] = [
            ind[0] + datetime.timedelta(minutes=mpl), 1]

    sz_onset_df.set_index('utc-datetime', inplace=True)

    return sz_onset_df


def get_indx_first_last_preictal_samples(labels_array):

    preictal_np_indx = np.argwhere(
        np.any(labels_array == 1, axis=1))[:, 0]
    diff_indx = np.diff(preictal_np_indx)

    last_preictal_np_indx = preictal_np_indx[np.reshape(
        np.argwhere(diff_indx != 1), (-1))]  # get last preictal point in sample
    first_preictal_np_indx = preictal_np_indx[np.reshape(
        np.argwhere(diff_indx != 1), (-1)) + 1]  # get first preictal point in sample

    last_preictal_np_indx = np.hstack(
        [last_preictal_np_indx, preictal_np_indx[-1]])  # add last preictal sample that was left out
    first_preictal_np_indx = np.hstack(
        [preictal_np_indx[0], first_preictal_np_indx])  # add first preictal sample that was left out

    assert len(first_preictal_np_indx) == len(last_preictal_np_indx)

    return first_preictal_np_indx, last_preictal_np_indx


def get_metadata(data_folder_path, patient_id):
    ''' Get metadata on available data files
    
    Parameters
    ---------- 
    data_folder_path : str
        Path to folder containing 'train_labels.csv'.
    patient_id : str
        Patient ID to filter data from the dataframe containing file metadata.
    sampling_frequency: int
        Frequency at which the data is stored in each file.
    
    Returns
    -------
    files_metadata : pd.DataFrame
        Dataframe containing metadata on the data files. Contains the following columns:
        - 'filepath' (str): fiepath to file (with 'data_folder_path' as parent folder).
        - 'first_timestamp' (int64): unix timestamp (in seconds) of the first data point in file.
        - 'total_duration' (int64): duration (in seconds) of the data within the file.
    sz_onsets : list
        Contains the unixe timstamp of the start of the files containing the onset of seizures
    ''' 

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

    return files_metadata, sz_onsets

    

def create_hdf5_dataset(files, dataset_filepath, sampling_frequency, features2extract):
    ''' Create hdf5 files containing a 'train' and a 'test' dataset.
    
    Parameters
    ---------- 
    files : list
        Contains the complete path for the files to include in the dataset.
    dataset_filepath : str
        Complete path to the hdf5 file.
    sampling_frequency : int
        Frequency at which the data is stored in each file.
    features2extract: dict
        Dict where keys correspond to the names of the channels in "channels_names" and the values are lists with the names of features. The features can be statistical, EDA event-based, or Hjorth features. Feature names should be the ones from the following list:
            - Statistical: "mean", "power", "std", "kurtosis", "skewness", "mean_1stdiff", "mean_2nddiff", "entropy".
            - EDA event-based: "SCR_amplitude", "SCR_peakcount", "mean_SCR_amplitude", "mean_SCR_risetime", "sum_SCR_amplitudes", "sum_SCR_risetimes", "SCR_AUC".
            - Hjorth: "hjorth_activity", "hjorth_mobility", "hjorth_complexity".
    
    Returns
    -------
    None
    ''' 
    real_time_vec, cpu_time_vec = [],[]
    real_time_bp, cpu_time_bp = [],[]

    for i, filepath in enumerate(files):

        try:
            timestamps_data, data, channels_names = read_and_segment(filepath, fs=sampling_frequency, decimate_factor=8)

            t1 = time.perf_counter(), time.process_time()
            _ = extract_features(data, channels_names, features2extract, sampling_frequency)
            t2 = time.perf_counter(), time.process_time()
            
            real_time_vec += [t2[0] - t1[0]]
            cpu_time_vec += [t2[1] - t1[1]]

            t1 = time.perf_counter(), time.process_time()
            _ = extract_features_bp(data, channels_names, features2extract, sampling_frequency)
            t2 = time.perf_counter(), time.process_time()
            
            real_time_bp += [t2[0] - t1[0]]
            cpu_time_bp += [t2[1] - t1[1]]
            
        except ZeroDivisionError:
            print(f'Not enough data on {filepath}\n')
            continue


        print(f"Processed {i+1}/{len(files)}", end="\r")
    
    print(f" Real time (vectorized): {np.mean(real_time_vec)*1000:.2f}+/-{np.std(real_time_vec)*1000:.2f} ms | (non-vectorized): {np.mean(real_time_bp)*1000:.2f}+/-{np.std(real_time_bp)*1000:.2f} ms")
    print(f" CPU time (vectorized): {np.mean(cpu_time_vec)*1000:.2f}+/-{np.std(cpu_time_vec)*1000:.2f} ms | (non-vectorized): {np.mean(cpu_time_bp)*1000:.2f}+/-{np.std(cpu_time_bp)*1000:.2f} ms")


def read_and_segment(filepath, decimate_factor=8, fs=128, sample_duration=60):  # 60s * 128Hz
    ''' Reads a file with 10min duration, splits into 60s samples, drops the samples with NaNs and returns a list with length correspoding to the number of samples and each element an array with
    dimension (sample length, # channels)

    Parameters
    ---------- 
    filepath : str
        Filepath to parquet file, with dataframe-like object, with 76800 rows (data points) and 9 columns (channels)
    decimate_factor : int, defaults to 8
        Decimates each sample by this factor (e.g. decimate_factor=8 -> new_fs=16Hz)
    fs : int, defaults to 128
        Frequency at which the data is stored. 
    sample_duration : int, defaults to 60s
        Desired duration for the samples (in seconds). 

    Returns
    -------
    raw_np_timestamps: array-like, shape (# samples, )
        Contains the start timestamp of each sample.
    raw_np_data: array-like, shape (# samples, # data points in sample, # channels)
        Data array.
    channels_names: list<str>
        List of strings, corresponding to the names of the channels.
    '''
    try:
        sample_length = sample_duration * fs

        raw_df = pd.read_parquet(filepath, engine="pyarrow")
        raw_np = np.array(np.split(raw_df.values, np.arange(
            sample_length, len(raw_df), sample_length), axis=0))
        
        channels_names = raw_df.columns.to_list()[1:]

        # removes sample if there is any NaN
        raw_np_noNaN = raw_np[~np.isnan(raw_np).any(axis=(1, 2))]

        raw_np_timestamps = raw_np_noNaN[:, 0, 0]
        raw_np_data = raw_np_noNaN[:, :, 1:]

        raw_np_data = scipy.signal.decimate(raw_np_data, decimate_factor, axis=1)

        return raw_np_timestamps, raw_np_data, channels_names

    except Exception as e:
        print(e)
        return None, None, None



def create_dataset(hdf, data):
    hdf.create_dataset('timestamps',
                       data=data[0], maxshape=(None,), dtype='int64') # float64??
    hdf.create_dataset(f'data', data=data[1], maxshape=(
        None, None, None,), dtype='float32')


def update_dataset(hdf, data):
    # check data validity
    if data != None:
        hdf[f'timestamps'].resize(
            (hdf[f'timestamps'].shape[0] + len(data[0])), axis=0)
        hdf[f'timestamps'][-len(data[0]):] = data[0]
        hdf[f'data'].resize(
            (hdf[f'data'].shape[0] + len(data[1])), axis=0)
        hdf[f'data'][-len(data[1]):] = data[1]
