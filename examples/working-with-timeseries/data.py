# built-in
import os
import datetime

# third-party
import h5py
import numpy as np
import pandas as pd
import scipy
import biosppy as bp
import scipy.signal


# local
from features import extract_features


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
        Contains the unix timestamp (in seconds) of the start of the files containing the onset of seizures
    '''

    # Convert file metadata and sz onsets into expected format
    train_labels_all = pd.read_csv(os.path.join(data_folder_path, 'train_labels.csv'))
    train_labels_all[['id', 'session', 'utc-datetime']] = train_labels_all['filepath'].str.split('/', expand=True)
    train_labels_all['utc-datetime'] = train_labels_all['utc-datetime'].str.strip('.parquet')

    if patient_id not in train_labels_all['id'].unique():
        raise ValueError(f'ID {patient_id} does not exist in dataset')

    train_labels = train_labels_all.loc[train_labels_all['id'] == patient_id, :]

    # Convert UTC datetime to Unix timestamp
    train_labels.loc[:, 'utc-datetime'] = pd.to_datetime(train_labels['utc-datetime'],
                                                         format='UTC-%Y_%m_%d-%H_%M_%S', utc=True, errors='coerce')
    files_metadata = pd.DataFrame({
        'filepath': train_labels.reset_index()['filepath'],
        # Convert to seconds
        'first_timestamp': train_labels.reset_index()['utc-datetime'].astype('datetime64[ns, UTC]').astype('int') // 10**9
    })
    files_metadata['total_duration'] = [10*60] * len(files_metadata)  # 10min in seconds

    # Get seizure onsets
    # mpl set to 10min to account for the pre-ictal labeling done in MSG2022
    sz_onset_df = get_sz_onset_df(train_labels.reset_index()[['utc-datetime', 'label']].to_numpy(), mpl=10)
    sz_onsets = (sz_onset_df.index.astype('int') // 10**9).tolist()

    return files_metadata, sz_onsets


def create_features_dataset(files, dataset_filepath, sampling_frequency, features2extract):
    ''' Create hdf5 file where each sample point corresponds to a set of features extracted.

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

    Raises
    ------
    ValueError : 
        When "features2extract" contains feature names that do not exist.
    '''

    with h5py.File(dataset_filepath, 'w') as hdf:

        lost_files = 0

        for i, filepath in enumerate(files):

            try:
                timestamps_data, data, channels_names, new_sampling_frequency = read_and_segment(
                    filepath, fs=sampling_frequency, decimate_factor=8)
                timestamps_data, data = preprocess(data, timestamps_data, channels_names, new_sampling_frequency)
                timestamps_data, data = quality_control(
                    data, timestamps_data, channels_names, sampling_frequency)  # TODO: NOT IMPLEMETED
                timestamps_data, data = extract_features(
                    data, timestamps_data, channels_names, features2extract, sampling_frequency)

                # transform first dimension (samples) into list
                data = np.split(data, data.shape[0], axis=0)
                data = [np.squeeze(x, axis=0) for x in data]
                timestamps_data = list(timestamps_data)

            except RuntimeError:
                print(f'Not enough data on {filepath}')
                lost_files += 1
                continue
            except FileNotFoundError:
                print(f'File {filepath} not found')
                lost_files += 1
                continue
            except AttributeError:
                print(f'No valid samples in {filepath}')
                lost_files += 1
                continue
            except ValueError:
                lost_files += 1
                print(f'File {filepath} could not be open.')
                continue

            dataset = (timestamps_data, data)

            if 'data' not in hdf.keys():
                create_dataset(hdf, dataset)
            else:
                update_dataset(hdf, dataset)

            print(f"Processed {i+1}/{len(files)}", end="\r")

        print(f'\n{lost_files} out of {len(files)} were ignored.')



def create_timeseries_dataset(files, dataset_filepath, sampling_frequency):
    ''' Create hdf5 file where each sample corresponds to a set of time series.

    Parameters
    ---------- 
    files : list
        Contains the complete path for the files to include in the dataset.
    dataset_filepath : str
        Complete path to the hdf5 file.
    sampling_frequency : int
        Frequency at which the data is stored in each file.

    Returns
    -------
    None
    '''

    with h5py.File(dataset_filepath, 'w') as hdf:

        lost_files = 0

        for i, filepath in enumerate(files):

            try:
                timestamps_data, data, channels_names, new_sampling_frequency = read_and_segment(
                    filepath, fs=sampling_frequency, decimate_factor=8)
                timestamps_data, data = preprocess(data, timestamps_data, channels_names, new_sampling_frequency)
                timestamps_data, data = quality_control(
                    data, timestamps_data, channels_names, sampling_frequency)  # TODO: NOT IMPLEMETED

                # transform first dimension (samples) into list
                data = np.split(data, data.shape[0], axis=0)
                data = [np.squeeze(x, axis=0) for x in data]
                timestamps_data = list(timestamps_data)

            except RuntimeError:
                print(f'Not enough data on {filepath}')
                lost_files += 1
                continue
            except FileNotFoundError:
                print(f'File {filepath} not found')
                lost_files += 1
                continue
            except AttributeError:
                print(f'No valid samples in {filepath}')
                lost_files += 1
                continue
            except ValueError:
                lost_files += 1
                print(f'File {filepath} could not be open.')
                continue

            dataset = (timestamps_data, data)

            if 'data' not in hdf.keys():
                create_dataset(hdf, dataset)
            else:
                update_dataset(hdf, dataset)

            print(f"Processed {i+1}/{len(files)}", end="\r")

        print(f'\n{lost_files} out of {len(files)} were ignored.')


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

    Raises
    ------
    RuntimeError :

    '''

    sample_length = sample_duration * fs

    try:
        raw_df = pd.read_parquet(filepath, engine="pyarrow")
    except FileNotFoundError:
        raise FileNotFoundError
    except:
        raise ValueError

    raw_np = np.array(np.split(raw_df.values, np.arange(
        sample_length, len(raw_df), sample_length), axis=0))

    channels_names = raw_df.columns.to_list()[1:]

    # removes sample if there is any NaN
    raw_np_noNaN = raw_np[~np.isnan(raw_np).any(axis=(1, 2))]

    if len(raw_np_noNaN) == 0:
        raise RuntimeError("No data in the file")

    raw_np_timestamps = raw_np_noNaN[:, 0, 0]
    raw_np_data = raw_np_noNaN[:, :, 1:]

    raw_np_data = scipy.signal.decimate(raw_np_data, decimate_factor, axis=1)

    return raw_np_timestamps, raw_np_data, channels_names, int(fs/decimate_factor)

    # except Exception as e:
    #     print(e)
    #     return None, None, None, None


def preprocess(samples, timestamps, channel_names, sampling_frequency):
    ''' Preprocess samples according to the type of signal (given by "channel_names") and remove unusable samples. Removes samples whose time series contain NaNs.  

    Parameters
        ---------- 
        samples: array-like, shape (#samples, #data points in sample, #channels)
            Data array.
        timestamps: array-like, shape (#samples, )
            Contains the start timestamp of each sample.
        channels_names: list<str>
            List of strings, corresponding to the names of the channels.
        sampling_frequency: int
            Frequency at which the data is presented along axis=1. 

    Returns
    -------
    timestamps: array-like, shape (#samples, )
        Contains the start timestamp of each sample.
    samples: array-like, shape (#samples, #data points in sample, #channels)
        Data array.
    '''

    if samples is None:
        return None

    preprocessed_data = []

    channel2function = {'acc_x': _acc_preprocess, 'acc_y': _acc_preprocess, 'acc_z': _acc_preprocess,
                        'acc_mag': _acc_preprocess, 'bvp': _bvp_preprocess, 'eda': _eda_preprocess, 'hr': _hr_preprocess, 'temp': _temp_preprocess}

    for channel_ind, channel_name in enumerate(channel_names):
        new_channel_data = channel2function[channel_name](samples[:, :, channel_ind], sampling_frequency)
        preprocessed_data += [new_channel_data]

    preprocessed_data = np.stack(preprocessed_data, axis=-1)

    # remove nan if existant
    valid_samples_indx = np.all(~np.isnan(preprocessed_data), axis=(1, 2))
    return timestamps[valid_samples_indx], preprocessed_data[valid_samples_indx, :, :]


def _acc_preprocess(array, sampling_frequency):
    """Internal method that applies a preprocessing methodology to accelerometer (ACC) samples in an array with shape (#samples, #data points in sample), and return as array with the same dimensions but potentially less #samples."""
    return array


def _bvp_preprocess(array, sampling_frequency):
    """Internal method that applies a preprocessing methodology to blood volume pulse (BVP) samples in an array with shape (#samples, #data points in sample), and return as array with the same dimensions but potentially less #samples."""
    # get filter coefficients
    b, a = scipy.signal.butter(
        N=4,
        Wn=[1, 7],
        fs=sampling_frequency,
        btype='bandpass'
    )
    return scipy.signal.filtfilt(b, a, array, axis=1)


def _eda_preprocess(array, sampling_frequency):
    """Internal method that applies a preprocessing methodology to electrodermal activity (EDA) samples in an array with shape (#samples, #data points in sample), and return as array with the same dimensions but potentially less #samples."""
    # get filter coefficients
    b, a = scipy.signal.butter(
        N=4,
        Wn=5,
        fs=sampling_frequency,
        btype='lowpass'
    )
    filtered = scipy.signal.filtfilt(b, a, array, axis=1)
    sm_size = int(0.75 * sampling_frequency)
    filtered, _ = _smoother(filtered, kernel="boxzen", size=sm_size, mirror=True)
    return filtered


def _hr_preprocess(array, sampling_frequency):
    """Internal method that applies a preprocessing methodology to heart rate (HR) samples in an array with shape (#samples, #data points in sample), and return as array with the same dimensions but potentially less #samples."""
    return array


def _temp_preprocess(array, sampling_frequency):
    """Internal method that applies a preprocessing methodology to temperature (TEMP) samples in an array with shape (#samples, #data points in sample), and return as array with the same dimensions but potentially less #samples."""
    return array


def _smoother(signal=None, kernel="boxzen", size=10, mirror=True, **kwargs):
    """Smooth a signal using an N-point moving average [MAvg]_ filter. Adapted from  BioSPPy.

    This implementation uses the convolution of a filter kernel with the input
    signal to compute the smoothed signal [Smit97]_.

    Availabel kernels: median, boxzen, boxcar, triang, blackman, hamming, hann,
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    kaiser (needs beta), gaussian (needs std), general_gaussian (needs power,
    width), slepian (needs width), chebwin (needs attenuation).

    Parameters
    ----------
    signal : array
        Signal to smooth.
    kernel : str, array, optional
        Type of kernel to use; if array, use directly as the kernel.
    size : int, optional
        Size of the kernel; ignored if kernel is an array.
    mirror : bool, optional
        If True, signal edges are extended to avoid boundary effects.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal.windows function.

    Returns
    -------
    signal : array
        Smoothed signal.
    params : dict
        Smoother parameters.

    Notes
    -----
    * When the kernel is 'median', mirror is ignored.

    References
    ----------
    .. [MAvg] Wikipedia, "Moving Average",
       http://en.wikipedia.org/wiki/Moving_average
    .. [Smit97] S. W. Smith, "Moving Average Filters - Implementation by
       Convolution", http://www.dspguide.com/ch15/1.htm, 1997

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    length = signal.shape[1]

    if isinstance(kernel, str):
        # check length
        if size > length:
            size = length - 1

        if size < 1:
            size = 1

        if kernel == "boxzen":
            # hybrid method
            # 1st pass - boxcar kernel
            aux, _ = _smoother(signal, kernel="boxcar", size=size, mirror=mirror)

            # 2nd pass - parzen kernel
            smoothed, _ = _smoother(aux, kernel="parzen", size=size, mirror=mirror)

            params = {"kernel": kernel, "size": size, "mirror": mirror}

            args = (smoothed, params)
            names = ("signal", "params")

            return bp.utils.ReturnTuple(args, names)

        else:
            win = bp.signals.tools._get_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)

        # check length
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")

        if size < 1:
            raise ValueError("Kernel size is smaller than 1.")

    else:
        raise TypeError("Unknown kernel type.")

    # convolve
    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal[:, 0][:, np.newaxis] * np.ones((len(signal), size)), signal, signal[:, -1][:, np.newaxis] * np.ones((len(signal), size))), axis=1
        )
        smoothed = scipy.signal.convolve(aux, w[np.newaxis, :], mode='same')
        smoothed = smoothed[:, size:-size]
    else:
        smoothed = scipy.signal.convolve(aux, w[np.newaxis, :], mode='same')

    # output
    params = {"kernel": kernel, "size": size, "mirror": mirror}
    params.update(kwargs)

    args = (smoothed, params)
    names = ("signal", "params")

    return bp.utils.ReturnTuple(args, names)


def quality_control(samples, timestamps, channel_names, sampling_frequency):
    ''' Quality control procedure according to the type of signal. Samples whose channels (any) are considered inadequate, are removed.   

    Parameters
        ---------- 
        samples: array-like, shape (#samples, #data points in sample, #channels)
            Data array.
        timestamps: array-like, shape (#samples, )
            Contains the start timestamp of each sample.
        channels_names: list<str>
            List of strings, corresponding to the names of the channels.
        sampling_frequency: int
            Frequency at which the data is presented along axis=1. 

    Returns
    -------
    timestamps: array-like, shape (#samples, )
        Contains the start timestamp of each sample.
    samples: array-like, shape (#samples, #data points in sample, #channels)
        Data array.
    '''
    if samples is None:
        return None

    quality = []

    channel2function = {'acc_x': _acc_quality, 'acc_y': _acc_quality, 'acc_z': _acc_quality,
                        'acc_mag': _acc_quality, 'bvp': _bvp_quality, 'eda': _eda_quality, 'hr': _hr_quality, 'temp': _temp_quality}

    for channel_ind, channel_name in enumerate(channel_names):
        new_channel_data = channel2function[channel_name](samples[:, :, channel_ind], sampling_frequency)
        quality += [new_channel_data]

    quality = np.concatenate(quality, axis=1)

    # remove nan if existant
    valid_samples_indx = np.all(~np.isnan(quality), axis=1)
    return timestamps[valid_samples_indx], samples[valid_samples_indx, :, :]


def _acc_quality(array, sampling_frequency):
    """Internal method that applies a quality control methodology to accelerometer (ACC) samples in an array with shape (#samples, #data points in sample), and return as array with shape (#samples, 1)."""
    return np.ones((len(array), 1))


def _bvp_quality(array, sampling_frequency):
    """Internal method that applies a quality control methodology to blood volume pulse (BVP) samples in an array with shape (#samples, #data points in sample), and return as array with shape (#samples, 1)."""
    return np.ones((len(array), 1))


def _eda_quality(array, sampling_frequency):
    """Internal method that applies a quality control methodology to elecrodermal activity (EDA) samples in an array with shape (#samples, #data points in sample), and return as array with shape (#samples, 1). Empirical thresholds based on Kleckner et al. (2018)."""
    return np.ones((len(array), 1))


def _hr_quality(array, sampling_frequency):
    """Internal method that applies a quality control methodology to heart rate (HR) samples in an array with shape (#samples, #data points in sample), and return as array with shape (#samples, 1)."""
    return np.ones((len(array), 1))


def _temp_quality(array, sampling_frequency):
    """Internal method that applies a quality control methodology to temperature (TEMP) samples in an array with shape (#samples, #data points in sample), and return as array with shape (#samples, 1). Empirical thresholds based on Kleckner et al. (2018)."""
    return np.ones((len(array), 1))


def create_dataset(hdf, data):
    hdf.create_dataset('timestamps',
                       data=data[0], maxshape=[None,], dtype='int64')
    hdf.create_dataset(f'data', data=data[1], maxshape=[None]+[d for d in data[1][0].shape], dtype='float32')


def update_dataset(hdf, data):
    # check data validity
    if data != None:
        hdf[f'timestamps'].resize(
            (hdf[f'timestamps'].shape[0] + len(data[0])), axis=0)
        hdf[f'timestamps'][-len(data[0]):] = data[0]
        hdf[f'data'].resize(
            (hdf[f'data'].shape[0] + len(data[1])), axis=0)
        hdf[f'data'][-len(data[1]):] = data[1]
