# -*- coding: utf-8 -*-
"""
sefef.labeling
--------------

This module contains functions to automatically label samples according to the desired pre-ictal duration and prediction latency.

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""


# built-in
import warnings

# third-party
import numpy as np


def add_annotations(h5dataset, sz_onsets_ts, preictal_duration=3600, prediction_latency=600):
    ''' Add "annotations", with shape (#samples, ) and dtype "bool", to HDF5 file object according to the variables "preictal_duration" and "prediction_latency". Annotations are either 0 (inter-ictal), or 1 (pre-ictal).

    Parameters
    ---------- 
    h5dataset : HDF5 file
        HDF5 file object with the following datasets:
        - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels). 
        - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
        - "sz_onsets": contains the Unix timestamps of the onsets of seizures (#sz_onsets, ). (optional)
    sz_onsets_ts : array-like, shape (#sz onsets, )
        Contains the unix timestamps (in seconds) of the onsets of seizures. 
    preictal_duration : int, defaults to 3600 (60min)
        Duration of the period (in seconds) that will be labeled as preictal, i.e. that we expect to contain useful information for the forecast
    prediction_latency : int, defaults to 600 (10min)
        Latency (in seconds) of the preictal period with regards to seizure onset.

    Returns
    -------
    None, but adds a dataset instance to the h5dataset file object.
    '''
    if 'timestamps' not in h5dataset.keys():
        raise KeyError('HDF5 file does not contain a "timestamps" dataset, which should contain the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).')

    if 'annotations' in h5dataset.keys():
        warnings.warn('Dataset already contains annotations. Skipping this step.')
        return None

    timestamps = h5dataset['timestamps'][()]
    labels = np.zeros(timestamps.shape, dtype='bool')

    for sz_ts in sz_onsets_ts:
        labels[np.where(np.logical_and(timestamps >= sz_ts-(preictal_duration +
                        prediction_latency), timestamps < sz_ts-prediction_latency))] = 1

    h5dataset.create_dataset("annotations", data=labels, dtype='bool')


def add_sz_onsets(h5dataset, sz_onsets_ts):
    ''' Add "sz_onsets", with shape (#seizures, ) and dtype "int64", to HDF5 file object, corresponding to the Unix timestamps of each seizure onset. 

    Parameters
    ---------- 
    h5dataset : HDF5 file
        HDF5 file object with the following datasets:
        - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels). 
        - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
        - "annotations": contains the annotations (aka labels) of each sample. (optional)
    sz_onsets_ts : array-like, shape (#sz onsets, )
        Contains the unix timestamps (in seconds) of the onsts of seizures. 

    Returns
    -------
    None, but adds a dataset instance to the h5dataset file object.
    '''

    if 'sz_onsets' in h5dataset.keys():
        warnings.warn('Dataset already contains the onsets of seizures. Skipping this step.')
        return None

    h5dataset.create_dataset("sz_onsets", data=sz_onsets_ts, dtype='int64')
