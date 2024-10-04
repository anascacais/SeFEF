# built-in
import datetime

# third-party
import numpy as np
import pandas as pd
import scipy


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



def read_and_segment(filepath, decimate_factor=8, fs=128, sample_duration=60):  # 60s * 128Hz
    ''' Reads a file with 10min duration, splits into 60s samples, drops the samples with NaNs and returns a list with length correspoding to the number of samples and each element an array with
    dimension (sample length, # channels)

    Parameters
    ---------- 
    filepath: str
        Filepath for parquet file, with dataframe-like object, with 76800 rows (data points) and 9 columns (channels)
    decimate_factor: int
        Decimates each sample by this factor (e.g. decimate_factor=8 -> new_fs=16Hz)

    Returns
    -------
    raw_list_timestamps: list<float>
        List (with length # samples) of the start timestamp of each sample.
    raw_list_data: list<np.array>
        List (with length # samples) of arrays with dimension (sample length, # channels).
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

        # transform first dimension (samples) into list
        raw_np_data = np.split(raw_np_data, raw_np_data.shape[0], axis=0)

        raw_list_timestamps = list(raw_np_timestamps)
        raw_list_data = [np.squeeze(x, axis=0) for x in raw_np_data]

        return raw_list_timestamps, raw_list_data, channels_names

    except Exception as e:
        return None, None, None