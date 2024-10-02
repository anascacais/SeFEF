# built-in
import datetime

# third-party
import numpy as np
import pandas as pd


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