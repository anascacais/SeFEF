# -*- coding: utf-8 -*-
"""
sefef.evaluation
----------------

This module contains functions to implement time-series cross validation (TSCV).

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""
# built-in
import copy

# third-party
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

# local
from .visualization import COLOR_PALETTE, hex_to_rgba


class TimeSeriesCV:
    ''' Implements time series cross validation (TSCV).

    Attributes
    ---------- 
    preictal_duration : int, defaults to 3600 (60min)
        Duration of the period (in seconds) that will be labeled as preictal, i.e. that we expect to contain useful information for the forecast
    prediction_latency : int, defaults to 600 (10min)
        Latency (in seconds) of the preictal period with regards to seizure onset.
    n_min_events_train : int, defaults to 3
        Minimum number of lead seizures to include in the train set. Should guarantee at least one lead seizure is left for testing.
    n_min_events_test : int, defaults to 1
        Minimum number of lead seizures to include in the test set. Should guarantee at least one lead seizure is left for testing.
    post_sz_interval : int
        Time interval (in seconds) after a lead seizure that should be included in the same set as the corresponding seizure. This time will be removed from the train set, along with the seizure onset and prediction_latency. 
    pre_lead_sz_interval : int
        Time interval (in seconds) free of seizures by which a seizure should be preceded to be considered a lead seizure.
    initial_train_duration : int, defaults to 1/3 of total recorded duration
        Set duration of train for initial split (in seconds). 
    test_duration : int, defaults to 1/2 of 'initial_train_duration'
        Set duration of test (in seconds). 
    method : str
        Method for TSCV - can be either 'expanding' or 'sliding'. Only 'expanding' is implemented atm.
    n_folds : int
        Number of folds for the TSCV, determined according to the attributes set by the user and available data.
    split_ind_ts : array-like, shape (n_folds, 3)
        Contains split timestamp indices (train_start_ts, test_start_ts, test_end_ts) for each fold. Is initiated as None and populated during 'split' method.

    Methods
    -------
    split(dataset, iteratively) : 
        Get timestamp indices to split data for time series cross-validation. 
        - The train set can be obtained by metadata.loc[train_start_ts : test_start_ts].
        - The test set can be obtained by metadata.loc[test_start_ts : test_end_ts].
    plot(dataset) :
        Plots the TSCV folds with the available data.
    iterate() : 
        Iterates over the TSCV folds and at each iteration returns a train set and a test set. 

    Raises
    -------
    ValueError :
        Raised whenever TSCV is not passible to be performed under the attributes set by the user and available data. 
    AttributeError :
        Raised when 'plot' is called before 'split'.
    '''

    def __init__(self, preictal_duration, prediction_latency, n_min_events_train=3, n_min_events_test=1, post_sz_interval=3600, pre_lead_sz_interval=14400, initial_train_duration=None, test_duration=None):
        self.preictal_duration = preictal_duration
        self.prediction_latency = prediction_latency

        self.post_sz_interval = post_sz_interval
        self.pre_lead_sz_interval = pre_lead_sz_interval

        self.n_min_events_train = n_min_events_train
        self.n_min_events_test = n_min_events_test
        self.initial_train_duration = initial_train_duration
        self.test_duration = test_duration
        self.method = 'expanding'

        self.n_folds = None
        self.split_ind_ts = None

    def split(self, dataset, iteratively=False, plot=False, extend_final_test_set=False):
        """ Get timestamp indices to split data for time series cross-validation. 
        - The train set would be given by metadata.loc[train_start_ts : test_start_ts].
        - The test set would be given by metadata.loc[test_start_ts : test_end_ts].

        Parameters:
        -----------
        dataset : Dataset
            Instance of Dataset.
        iteratively : bool, defaults to False
            If the split is meant to return the timestamp indices for each fold iteratively (True) or to simply update 'split_ind_ts' (False). 
        plot : bool, defaults to False
            If a diagram illustrating the TSCV should be shown at the end. 'iteratively' cannot be set to True
        extend_final_test_set : bool
            Whether to extend test set in final fold to include all data or keep test duration approximately the same across folds.

        Returns:
        --------
        train_start_ts : int
            Timestamp index for the start of the train set.
        test_start_ts : int
            Timestamp index for the start of the test set (and end of train set).
        test_end_ts : int
            Timestamp index for the end of the test set.
        """

        dataset_lead_sz = self._get_lead_sz_dataset(dataset)

        if self.initial_train_duration is None:
            total_recorded_duration = dataset_lead_sz.files_metadata['total_duration'].sum()
            if total_recorded_duration == 0:
                raise ValueError(f"Dataset is empty.")
            self.initial_train_duration = (1/3) * total_recorded_duration

        if self.test_duration is None:
            self.test_duration = (1/2) * self.initial_train_duration

        # Check basic conditions
        if dataset_lead_sz.files_metadata['total_duration'].sum() < self.initial_train_duration + self.test_duration:
            raise ValueError(
                f"Dataset does not contain enough data to do this split. Just give up (or decrease 'initial_train_duration' ({self.initial_train_duration}) and/or 'test_duration' ({self.test_duration})).")

        if dataset_lead_sz.metadata['sz_onset'].sum() < self.n_min_events_train + self.n_min_events_test:
            raise ValueError(
                f"Dataset does not contain the minimum number of events. Just give up (or change the value of 'n_min_events_train' ({self.n_min_events_train}) or 'n_min_events_test' ({self.n_min_events_test})).")

        # Get index for initial split
        initial_cutoff_ts = self._get_cutoff_ts(dataset_lead_sz)
        initial_cutoff_ts = self._check_criteria_initial_split(dataset_lead_sz, initial_cutoff_ts)
        print('\n')

        if iteratively:
            if plot:
                raise ValueError("The variables 'iteratively' and 'plot' cannot both be set to True.")
            if extend_final_test_set:
                raise NotImplementedError(
                    "Setting both 'iteratively' and 'extend_final_test_set' to True is currently not supported.")
            return self._expanding_window_split(dataset_lead_sz, initial_cutoff_ts)
        else:
            for _ in self._expanding_window_split(dataset_lead_sz, initial_cutoff_ts):
                pass
            if extend_final_test_set:
                self.split_ind_ts[-1, 2] = dataset_lead_sz.metadata.index[-1]
            if plot:
                self.plot(dataset_lead_sz)
            return None

    def _expanding_window_split(self, dataset, initial_cutoff_ts):
        """Internal method for expanding window cross-validation."""

        after_train_set = dataset.metadata.loc[initial_cutoff_ts:]
        train_start_ts = dataset.metadata.index[0]
        test_start_ts = initial_cutoff_ts.copy()

        test_end_ts = 0
        split_ind_ts = []

        while test_end_ts <= dataset.metadata.iloc[-1].name:
            if test_end_ts != 0:
                after_train_set = dataset.metadata.loc[test_end_ts:]
                test_start_ts = test_end_ts

            try:
                test_end_ts = after_train_set.index[after_train_set['total_duration'].cumsum() >= self.test_duration].tolist()[
                    0]
                test_end_ts = self._check_criteria_split(after_train_set, test_end_ts)
                split_ind_ts += [[train_start_ts, test_start_ts, test_end_ts]]
            except IndexError:
                break
            yield train_start_ts, test_start_ts, test_end_ts

        self.split_ind_ts = np.array(split_ind_ts)
        self.n_folds = len(self.split_ind_ts)
        print('\n')

    def _sliding_window_split(self):
        """Internal method for sliding window cross-validation."""
        pass

    def _get_cutoff_ts(self, dataset):
        """Internal method for getting the first iteration of the cutoff timestamp based on 'self.initial_train_duration'."""
        cutoff_ts = dataset.metadata.index[dataset.metadata['total_duration'].cumsum() > self.initial_train_duration].tolist()[
            0]
        return cutoff_ts

    def _get_lead_seizures(self, ts_seizures):
        '''Internal method that, from metadata returns the timestamps of lead seizures, defined by a seizure preceded by "lead_sz_pre_interval".'''
        ts_lead_seizures = ts_seizures.copy()
        while not all(np.diff(ts_lead_seizures) >= self.pre_lead_sz_interval):
            ts_lead_seizures = ts_lead_seizures[np.concat(
                (np.array([True]), (np.diff(ts_lead_seizures) >= self.pre_lead_sz_interval)))]
        return ts_lead_seizures

    def _get_lead_sz_dataset(self, dataset):
        '''Internal method that returns a copy of the original dataset without the non-lead seizures.'''
        dataset_lead_sz = copy.deepcopy(dataset)
        ts_lead_sz = self._get_lead_seizures(dataset.metadata[dataset.metadata['sz_onset'] == 1].index.to_numpy())
        ts_all_sz = dataset.metadata[dataset.metadata['sz_onset'] == 1].index.to_numpy()
        dataset_lead_sz.metadata.loc[ts_all_sz[~np.any(
            ts_all_sz[:, np.newaxis] == ts_lead_sz[np.newaxis, :], axis=1)], 'sz_onset'] = 0
        return dataset_lead_sz

    def _check_criteria_initial_split(self, dataset, initial_cutoff_ts):
        """Internal method for iterating the initial cutoff timestamp in order to respect the condition on the minimum number of seizures."""

        criteria_check = [False] * 2

        initial_cutoff_ind = dataset.metadata.index.get_loc(initial_cutoff_ts)

        t = 0

        while not all(criteria_check):
            initial_train_set = dataset.metadata.iloc[:initial_cutoff_ind]
            after_train_set = dataset.metadata.iloc[initial_cutoff_ind:]

            # Criteria 1: min number of events in train
            criteria_check[0] = ((initial_train_set['sz_onset'].sum() >= self.n_min_events_train) &
                                 (self._check_if_preictal(initial_train_set) >= self.n_min_events_train))
            # Criteria 2: min number of events in test
            criteria_check[1] = ((after_train_set['sz_onset'].sum() >= self.n_min_events_test) &
                                 (self._check_if_preictal(after_train_set) >= self.n_min_events_test))

            if not all(criteria_check):
                print(
                    f"Initial split: failed criteria {[i+1 for i, val in enumerate(criteria_check) if not val]} (trial {t+1})", end='\r')

                if (not criteria_check[0]) and (not criteria_check[1]):
                    raise ValueError(
                        "Dataset does not comply with the conditions for this split. Just give up (or decrease 'n_min_events_train', 'initial_train_duration', and/or 'test_duration').")
                elif not criteria_check[0]:
                    initial_cutoff_ind += 1
                elif not criteria_check[1]:
                    initial_cutoff_ind -= 1

            t += 1

        # Check if there's enough data left for at least one test set
        if after_train_set['total_duration'].sum() < self.test_duration:
            raise ValueError(
                f"Dataset does not comply with the conditions for this split. Just give up (or decrease 'n_min_events_train' ({self.n_min_events_train}), 'initial_train_duration' ({self.initial_train_duration}), and/or 'test_duration' ({self.test_duration})).")

        # Account for "post_sz_interval" if split is immediately after a seizure
        if dataset.metadata.iloc[initial_cutoff_ind-1]['sz_onset'] == 1:
            ts_post_lead_sz = dataset.metadata.iloc[initial_cutoff_ind-1].name + self.post_sz_interval
            initial_cutoff_ind = np.where(dataset.metadata.index <= ts_post_lead_sz)[0][-1] + 1

        return dataset.metadata.iloc[initial_cutoff_ind].name

    def _check_if_preictal(self, metadata):
        '''Internal method that counts the number of seizure onsets for which there exist preictal samples.'''

        preictal_starts = metadata[metadata['sz_onset'] == 1].index.to_numpy(
        ) - self.preictal_duration - self.prediction_latency
        preictal_ends = metadata[metadata['sz_onset'] == 1].index.to_numpy() - self.prediction_latency

        # For each seizure onset, count number of samples within preictal period
        nb_preictal_samples = np.sum(np.logical_and(
            metadata.index.to_numpy()[:, np.newaxis] >= preictal_starts[np.newaxis, :],
            metadata.index.to_numpy()[:, np.newaxis] < preictal_ends[np.newaxis, :],
        ), axis=0)

        return np.count_nonzero(nb_preictal_samples)

    def _check_criteria_split(self, metadata, cutoff_ts):
        """Internal method for iterating the cutoff timestamp for n>1 folds in order to respect the condition on the minimum number of seizures in test."""

        criteria_check = [False] * 2
        cutoff_ind = metadata.index.get_loc(cutoff_ts)

        t = 0

        while not all(criteria_check):
            test_set = metadata.iloc[:cutoff_ind]
            # Criteria 1: Check if there's enough data left for a test set
            criteria_check[0] = cutoff_ind <= len(metadata)
            # Criteria 2: min number of events in test
            criteria_check[1] = ((test_set['sz_onset'].sum() >= self.n_min_events_test) &
                                 (self._check_if_preictal(test_set) >= self.n_min_events_test))

            if not all(criteria_check):
                print(
                    f"Initial split: failed criteria {[i+1 for i, val in enumerate(criteria_check) if not val]} (trial {t+1})", end='\r')

                if not criteria_check[0]:
                    return metadata.iloc[cutoff_ind].name
                elif not criteria_check[1]:
                    cutoff_ind += 1

            t += 1

        # Account for "post_sz_interval" if split is immediately after a seizure
        if metadata.iloc[cutoff_ind-1]['sz_onset'] == 1:
            ts_post_lead_sz = metadata.iloc[cutoff_ind-1].name + self.post_sz_interval
            cutoff_ind = np.where(metadata.index <= ts_post_lead_sz)[0][-1] + 1
        elif metadata.iloc[cutoff_ind]['sz_onset'] == 1:
            ts_post_lead_sz = metadata.iloc[cutoff_ind].name + self.post_sz_interval
            cutoff_ind = np.where(metadata.index <= ts_post_lead_sz)[0][-1] + 1

        if cutoff_ind == len(metadata):
            cutoff_ind -= 1

        return metadata.iloc[cutoff_ind].name

    def plot(self, dataset, folder_path=None, filename=None, mode='lines'):
        ''' Plots the TSCV folds with the available data.

        Parameters
        ---------- 
        dataset : Dataset
            Instance of Dataset.
        mode : str
            Trace scatter mode ("lines" or "markers"), for sparse data, "markers" is a more suitable option, despite being heavier to plot.
        '''
        if self.split_ind_ts is None:
            raise AttributeError(
                "Object has no attribute 'split_ind_ts'. Make sure the 'split' method has been run beforehand.")

        fig = go.Figure()

        file_duration = dataset.metadata['total_duration'].iloc[0]

        for ifold in range(self.n_folds):

            train_set = dataset.metadata.loc[self.split_ind_ts[ifold, 0]: self.split_ind_ts[ifold, 1]]
            test_set = dataset.metadata.loc[self.split_ind_ts[ifold, 1]: self.split_ind_ts[ifold, 2]]

            # handle missing data between files
            train_set = self._handle_missing_data(train_set, ifold+1, file_duration)
            test_set = self._handle_missing_data(test_set, ifold+1, file_duration)

            # add existant data
            fig.add_trace(self._get_scatter_plot(
                train_set, color=COLOR_PALETTE[0], mode=mode, name='Train', showlegend=(ifold == 0)))
            fig.add_trace(self._get_scatter_plot(
                test_set, color=COLOR_PALETTE[1], mode=mode, name='Test', showlegend=(ifold == 0)))

            # add seizures
            ts_lead_sz = self._get_lead_seizures(train_set[train_set['sz_onset'] == 1].index.to_numpy())
            fig.add_trace(self._get_scatter_plot_sz(
                train_set.loc[ts_lead_sz],
                color=COLOR_PALETTE[0]
            ))
            ts_non_lead_sz = train_set[train_set['sz_onset'] == 1].index.to_numpy()[~np.any(
                train_set[train_set['sz_onset'] == 1].index.to_numpy()[:, np.newaxis] == ts_lead_sz[np.newaxis, :], axis=1)]
            fig.add_trace(self._get_scatter_plot_sz(
                train_set.loc[ts_non_lead_sz],
                color=COLOR_PALETTE[0],
                opacity=0.5
            ))
            fig.add_trace(self._get_scatter_plot_sz(
                test_set[test_set['sz_onset'] == 1],
                color=COLOR_PALETTE[1]
            ))

            # fig.add_trace(self._get_dummy_scatter(color=COLOR_PALETTE[2], showlegend=(ifold == 0)))

        # Config plot layout
        fig.update_yaxes(
            title='TSCV folds',
            gridcolor='lightgrey',
            autorange='reversed',
            tickvals=list(range(1, self.n_folds+1)),
            ticktext=[f'Fold {i}  ' for i in range(1, self.n_folds+1)],
            tickfont=dict(size=12),
        )
        fig.update_xaxes(title='Time')
        fig.update_layout(
            title='Time Series Cross Validation',
            # showlegend=False,
            plot_bgcolor='white')
        fig.show()

        if folder_path is not None:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            fig.write_image(os.path.join(folder_path, filename))

    def _handle_missing_data(self, metadata_no_nan, ind, duration):
        """Internal method that updates the received dataset with NaN corresponding to where there are no files containing data."""

        metadata = metadata_no_nan.copy()
        metadata.index = pd.to_datetime(metadata.index, unit='s')
        metadata.insert(0, 'data', ind)
        metadata = metadata.asfreq(freq=f'{duration}s')
        return metadata

    def _get_scatter_plot_sz(self, metadata, color, opacity=1, showlegend=False):
        """Internal method that returns a marker-scatter-plot where sz onsets exist."""
        return go.Scatter(
            x=metadata.index,
            y=metadata.data-0.1,
            showlegend=showlegend,
            name='Seizure',
            mode='markers+text',
            text=['ϟ'] * len(metadata),
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            textfont=dict(
                size=16,
                color='rgba' + str(hex_to_rgba(
                    h=color, alpha=opacity
                ))
            )
        )

    def _get_dummy_scatter(self, color, showlegend, opacity=1):
        """Internal method that returns a scatter plot without data only to create a legend item with neutral color for sz"""
        return go.Scatter(
            x=[None],
            y=[None],
            showlegend=showlegend,
            name='Seizure',
            mode="markers+text",
            text=['ϟ'],
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            textfont=dict(
                size=16,
                color='rgba' + str(hex_to_rgba(
                    h=color, alpha=opacity
                ))
            )
        )

    def _get_scatter_plot(self, metadata, color, mode, name, showlegend):
        """Internal method that returns a line-scatter-plot where data exists."""
        return go.Scatter(
            x=metadata.index,
            y=metadata.data,
            name=name,
            showlegend=showlegend,
            mode=mode,
            marker={
                'color': 'rgba' + str(hex_to_rgba(
                    h=color,
                    alpha=1
                )),
                'size': 5
            },
            line={
                'color': 'rgba' + str(hex_to_rgba(
                    h=color,
                    alpha=1
                )),
                'width': 5
            }
        )

    def iterate(self, h5dataset, remove_non_preictal_interictal_samples=True):
        ''' Iterates over the TSCV folds and at each iteration returns a train set and a test set. 

        Parameters
        ---------- 
        h5dataset : HDF5 file
            HDF5 file object with the following datasets:
            - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels) 
            - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
            - "annotations": contains the labels (0: interictal, 1: preictal) for each sample in the "data" dataset, with shape (#samples, ).
            - "sz_onsets": contains the Unix timestamps of the onsets of seizures (#sz_onsets, ). 
        remove_non_preictal_interictal_samples : bool
            Whether to remove samples that are neither preictal or interical, i.e. samples containing the onsets of seizures, as well as the intervals corrsponding to "prediction_latency" and "lead_sz_post_interval". 

        Returns
        -------
        tuple: 
            - ((train_data, train_annotations, train_timestamps), (test_data, test_sz_onsets, test_timestamps))
            - Where:
                - "[]_data": A slice of "h5dataset["data"]", with shape (#samples, embedding shape), e.g. (#samples, #features) or (#samples, sample duration, #channels), and dtype "float32".
                - "[]_annotations": A slice of "h5dataset["annotations"]", with shape (#samples, ) and dtype "bool".
                - "[]_sz_onsets": A slice of "h5dataset["sz_onsets"]", with shape (#sz onsets, ) and dtype "int64". 
                - "[]_timestamps": A slice of "h5dataset["timestamps"]", with shape (#samples, ) and dtype "int64". 
        '''
        timestamps = h5dataset['timestamps'][()]
        sz_onsets = h5dataset['sz_onsets'][()]
        annotations = h5dataset['annotations'][()]

        for train_start_ts, test_start_ts, test_end_ts in self.split_ind_ts:

            train_indx = np.where(np.logical_and(timestamps >= train_start_ts, timestamps < test_start_ts))
            test_indx = np.where(np.logical_and(timestamps >= test_start_ts, timestamps < test_end_ts))

            train_sz_indx = np.where(np.logical_and(sz_onsets >= train_start_ts, sz_onsets < test_start_ts))
            test_sz_indx = np.where(np.logical_and(sz_onsets >= test_start_ts, sz_onsets < test_end_ts))
            ts_train_lead_sz = self._get_lead_seizures(sz_onsets[train_sz_indx])

            # Remove from train the samples that are within the start of a preictal period and the end of "post_sz_interval"
            if remove_non_preictal_interictal_samples:
                ts_train = h5dataset['timestamps'][train_indx]
                not_relevant_indx = np.where(np.any(np.logical_and(ts_train[:, np.newaxis] >= ts_train_lead_sz[np.newaxis, :] -
                                             self.prediction_latency, ts_train[:, np.newaxis] <= ts_train_lead_sz[np.newaxis, :] + self.post_sz_interval), axis=1))[0]

                ts_train_non_lead_sz = sz_onsets[train_sz_indx][~np.any(
                    sz_onsets[train_sz_indx][:, np.newaxis] == ts_train_lead_sz[np.newaxis, :], axis=1)]
                ts_train_interictal = timestamps[train_indx][annotations[train_indx] == 0]
                not_relevant_indx = np.unique(np.concat((
                    not_relevant_indx,
                    np.where(np.any(ts_train[:, np.newaxis] == ts_train_interictal[np.where(np.any(np.logical_and(ts_train_interictal[:, np.newaxis] >= ts_train_non_lead_sz[np.newaxis, :] -
                             self.prediction_latency, ts_train_interictal[:, np.newaxis] <= ts_train_non_lead_sz[np.newaxis, :] + self.post_sz_interval), axis=1))][np.newaxis, :], axis=1))[0]
                )))

                train_indx = (np.setdiff1d(train_indx, not_relevant_indx),)

            yield (
                (h5dataset['data'][train_indx], h5dataset['annotations'][train_indx],
                 h5dataset['timestamps'][train_indx], ts_train_lead_sz),
                (h5dataset['data'][test_indx], h5dataset['annotations'][test_indx],
                 h5dataset['timestamps'][test_indx], sz_onsets[test_sz_indx])
            )

    def get_TSCV_fold(self, h5dataset, ifold, remove_non_preictal_interictal_samples=True):
        ''' Returns a train set and a test set  from corresponding TSCV fold. 

        Parameters
        ---------- 
        h5dataset : HDF5 file
            HDF5 file object with the following datasets:
            - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels) 
            - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
            - "annotations": contains the labels (0: interictal, 1: preictal) for each sample in the "data" dataset, with shape (#samples, ).
            - "sz_onsets": contains the Unix timestamps of the onsets of seizures (#sz_onsets, ). 
        ifold : int
            Index corresponding to TSCV fold.
        remove_non_preictal_interictal_samples : bool
            Whether to remove samples that are neither preictal or interical, i.e. samples containing the onsets of seizures, as well as the intervals corrsponding to "prediction_latency" and "lead_sz_post_interval". 

        Returns
        -------
        tuple: 
            - ((train_data, train_annotations, train_timestamps, train_sz_onsets), (test_data, test_annotations, test_timestamps, test_sz_onsets))
            - Where:
                - "[]_data": A slice of "h5dataset["data"]", with shape (#samples, embedding shape), e.g. (#samples, #features) or (#samples, sample duration, #channels), and dtype "float32".
                - "[]_annotations": A slice of "h5dataset["annotations"]", with shape (#samples, ) and dtype "bool".
                - "[]_timestamps": A slice of "h5dataset["timestamps"]", with shape (#samples, ) and dtype "int64". 
                - "[]_sz_onsets": A slice of "h5dataset["sz_onsets"]", with shape (#sz onsets, ) and dtype "int64". 
        '''
        timestamps = h5dataset['timestamps'][()]
        sz_onsets = h5dataset['sz_onsets'][()]
        annotations = h5dataset['annotations'][()]

        train_start_ts, test_start_ts, test_end_ts = self.split_ind_ts[ifold, :].tolist()

        train_indx = np.where(np.logical_and(timestamps >= train_start_ts, timestamps < test_start_ts))
        test_indx = np.where(np.logical_and(timestamps >= test_start_ts, timestamps < test_end_ts))

        train_sz_indx = np.where(np.logical_and(sz_onsets >= train_start_ts, sz_onsets < test_start_ts))
        test_sz_indx = np.where(np.logical_and(sz_onsets >= test_start_ts, sz_onsets < test_end_ts))
        ts_train_lead_sz = self._get_lead_seizures(sz_onsets[train_sz_indx])

        # Remove from train the samples that are within the start of a preictal period and the end of "post_sz_interval"
        if remove_non_preictal_interictal_samples:
            ts_train = h5dataset['timestamps'][train_indx]
            not_relevant_indx = np.where(np.any(np.logical_and(ts_train[:, np.newaxis] >= ts_train_lead_sz[np.newaxis, :] -
                                         self.prediction_latency, ts_train[:, np.newaxis] <= ts_train_lead_sz[np.newaxis, :] + self.post_sz_interval), axis=1))[0]

            ts_train_non_lead_sz = sz_onsets[train_sz_indx][~np.any(
                sz_onsets[train_sz_indx][:, np.newaxis] == ts_train_lead_sz[np.newaxis, :], axis=1)]
            ts_train_interictal = timestamps[train_indx][annotations[train_indx] == 0]
            not_relevant_indx = np.unique(np.concat((
                not_relevant_indx,
                np.where(np.any(ts_train[:, np.newaxis] == ts_train_interictal[np.where(np.any(np.logical_and(ts_train_interictal[:, np.newaxis] >= ts_train_non_lead_sz[np.newaxis, :] -
                         self.prediction_latency, ts_train_interictal[:, np.newaxis] <= ts_train_non_lead_sz[np.newaxis, :] + self.post_sz_interval), axis=1))][np.newaxis, :], axis=1))[0]
            )))

            train_indx = (np.setdiff1d(train_indx, not_relevant_indx),)

        return (
            (h5dataset['data'][train_indx], h5dataset['annotations'][train_indx],
                h5dataset['timestamps'][train_indx], ts_train_lead_sz),
            (h5dataset['data'][test_indx], h5dataset['annotations'][test_indx],
                h5dataset['timestamps'][test_indx], sz_onsets[test_sz_indx])
        )


class Dataset:
    ''' Create a Dataset with metadata on the data that will be used for training and testing

    Attributes
    ---------- 
    files_metadata: pd.DataFrame
        Input DataFrame with the following columns:
        - 'filepath' (str): Path to each file containing data.
        - 'first_timestamp' (int64): The Unix-time timestamp (in seconds) of the first sample of each file.
        - 'total_duration' (int64): Total duration of file in seconds (equivalent to #samples * sampling_frequency)
        It is expected that data within each file is non-overlapping in time and that there are no time gaps between samples in the file. 
    sz_onsets: np.array
        Contains the Unix-time timestamps (in seconds) corresponding to the onsets of seizures.
    sampling_frequency: int
        Frequency at which the data is stored in each file.
    '''

    def __init__(self, files_metadata, sz_onsets):
        self.files_metadata = files_metadata.astype({'first_timestamp': 'int64', 'total_duration': 'int64'})
        self.sz_onsets = np.array(sz_onsets)

        self.metadata = self._get_metadata()
        self.metadata = self.metadata.astype({'filepath': str})

    def _get_metadata(self):
        """Internal method that updates 'self.metadata' by placing each seizure onset within an acquisition file."""

        timestamps_file_start = self.files_metadata['first_timestamp'].to_numpy()
        timestamps_file_end = (self.files_metadata['first_timestamp'] +
                               self.files_metadata['total_duration']).to_numpy()

        # identify seizures within existant files
        sz_onset_indx = np.argwhere((self.sz_onsets[:, np.newaxis] >= timestamps_file_start[np.newaxis, :]) & (
            self.sz_onsets[:, np.newaxis] < timestamps_file_end[np.newaxis, :]))

        files_metadata = self.files_metadata.copy()
        files_metadata['sz_onset'] = 0
        files_metadata.loc[sz_onset_indx[:, 1], 'sz_onset'] = 1

        # identify seizures outside of existant files
        sz_onset_indx = np.argwhere(~np.any(((self.sz_onsets[:, np.newaxis] >= timestamps_file_start[np.newaxis, :]) & (
            self.sz_onsets[:, np.newaxis] < timestamps_file_end[np.newaxis, :])), axis=1)).flatten()
        if len(sz_onset_indx) != 0:
            sz_onsets = pd.DataFrame({'first_timestamp': self.sz_onsets[sz_onset_indx], 'sz_onset': [
                                     1]*len(sz_onset_indx)}, dtype='int64')
            files_metadata = pd.merge(files_metadata.reset_index(), sz_onsets.reset_index(),
                                      on='first_timestamp', how='outer', suffixes=('_df1', '_df2'))
            files_metadata['sz_onset'] = files_metadata['sz_onset_df1'].combine_first(
                files_metadata['sz_onset_df2']).fillna(0).astype('int64')
            files_metadata['total_duration'] = files_metadata['total_duration'].fillna(0).astype('int64')

        files_metadata.set_index(pd.Index(files_metadata['first_timestamp'].to_numpy(), dtype='int64'), inplace=True)
        files_metadata = files_metadata.loc[:, ['filepath', 'total_duration', 'sz_onset']]

        try:
            files_metadata = pd.concat((
                files_metadata, pd.DataFrame([[np.nan, 0, 0]], columns=files_metadata.columns, index=pd.Series(
                    [files_metadata.iloc[-1].name+files_metadata.iloc[-1]['total_duration']], dtype='int64')),
            ), ignore_index=False)  # add empty row at the end for indexing
        except IndexError:
            pass
        return files_metadata
