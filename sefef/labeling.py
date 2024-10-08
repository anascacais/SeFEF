# third-party 
import numpy as np


def add_annotations(h5dataset, sz_onsets_ts, preictal_duration=3600, prediction_latency=600):
    ''' Add "annotations", with shape (#samples, ) and dtype "bool", to HDF5 file object according to the variables "preictal_duration" and "prediction_latency".
    
    Parameters
    ---------- 
    h5dataset : HDF5 file
        HDF5 file object with the following datasets:
        - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels). 
        - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
    sz_onsets_ts : array-like, shape (#sz onsets, )
        Contains the unix timestamps (in seconds) of the onsts of seizures. 
    preictal_duration : int, defaults to 3600 (60min)
        Duration of the period (in seconds) that will be labeled as preictal, i.e. that we expect to contain useful information for the forecast
    prediction_latency : int, defaults to 600 (10min)
        Latency (in seconds) of the preictal period with regards to seizure onset.
    ''' 

    timestamps = h5dataset['timestamps'][()]
    labels = np.zeros(timestamps.shape, dtype='bool') 

    for sz_ts in sz_onsets_ts:
        labels[np.where(np.logical_and(timestamps >= sz_ts-(preictal_duration+prediction_latency), timestamps < sz_ts-prediction_latency))] = 1

    h5dataset.create_dataset("annotations", data=labels, dtype='bool')

