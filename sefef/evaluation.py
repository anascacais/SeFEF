# -*- coding: utf-8 -*-
"""
sefef.evaluation
----------------

This module contains functions to implement time-series cross validation (TSCV).

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""

# third-party
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# local
from .visualization import COLOR_PALETTE, hex_to_rgba


class TimeSeriesCV:
    ''' Implements time series cross validation (TSCV).

    Attributes
    ---------- 
    n_min_events_train : int, defaults to 3
        Minimum number of lead seizures to include in the train set. Should guarantee at least one lead seizure is left for testing.
    n_min_events_test : int, defaults to 1
        Minimum number of lead seizures to include in the test set. Should guarantee at least one lead seizure is left for testing.
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

    def __init__(self, n_min_events_train=3, n_min_events_test=1, initial_train_duration=None, test_duration=None):
        self.n_min_events_train = n_min_events_train
        self.n_min_events_test = n_min_events_test
        self.initial_train_duration = initial_train_duration
        self.test_duration = test_duration
        self.method = 'expanding'

        self.n_folds = None
        self.split_ind_ts = None

    def split(self, dataset, iteratively=False, plot=False):
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

        Returns:
        --------
        train_start_ts : int
            Timestamp index for the start of the train set.
        test_start_ts : int
            Timestamp index for the start of the test set (and end of train set).
        test_end_ts : int
            Timestamp index for the end of the test set.
        """

        if self.initial_train_duration is None:
            total_recorded_duration = dataset.files_metadata['total_duration'].sum()
            if total_recorded_duration == 0:
                raise ValueError(f"Dataset is empty.")
            self.initial_train_duration = (1/3) * total_recorded_duration

        if self.test_duration is None:
            self.test_duration = (1/2) * self.initial_train_duration

        # Check basic conditions
        if dataset.files_metadata['total_duration'].sum() < self.initial_train_duration + self.test_duration:
            raise ValueError(
                f"Dataset does not contain enough data to do this split. Just give up (or decrease 'initial_train_duration' ({self.initial_train_duration}) and/or 'test_duration' ({self.test_duration})).")

        if dataset.metadata['sz_onset'].sum() < self.n_min_events_train + self.n_min_events_test:
            raise ValueError(
                f"Dataset does not contain the minimum number of events. Just give up (or change the value of 'n_min_events_train' ({self.n_min_events_train}) or 'n_min_events_test' ({self.n_min_events_test})).")

        # Get index for initial split
        initial_cutoff_ts = self._get_cutoff_ts(dataset)
        initial_cutoff_ts = self._check_criteria_initial_split(dataset, initial_cutoff_ts)
        print('\n')

        if iteratively:
            if plot:
                raise ValueError("The variables 'iteratively' and 'plot' cannot both be set to True.")
            return self._expanding_window_split(dataset, initial_cutoff_ts)
        else:
            for _ in self._expanding_window_split(dataset, initial_cutoff_ts):
                pass
            if plot:
                self.plot(dataset)
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

    def _check_criteria_initial_split(self, dataset, initial_cutoff_ts):
        """Internal method for iterating the initial cutoff timestamp in order to respect the condition on the minimum number of seizures."""

        criteria_check = [False] * 2

        initial_cutoff_ind = dataset.metadata.index.get_loc(initial_cutoff_ts)

        t = 0

        while not all(criteria_check):
            initial_train_set = dataset.metadata.iloc[:initial_cutoff_ind]
            after_train_set = dataset.metadata.iloc[initial_cutoff_ind:]

            # Criteria 1: min number of events in train
            criteria_check[0] = initial_train_set['sz_onset'].sum() >= self.n_min_events_train
            # Criteria 2: min number of events in test
            criteria_check[1] = after_train_set['sz_onset'].sum() >= self.n_min_events_test

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

        return dataset.metadata.iloc[initial_cutoff_ind].name

    def _check_criteria_split(self, dataset, cutoff_ts):
        """Internal method for iterating the cutoff timestamp for n>1 folds in order to respect the condition on the minimum number of seizures in test."""

        criteria_check = [False] * 2
        cutoff_ind = dataset.index.get_loc(cutoff_ts)

        t = 0

        while not all(criteria_check):
            test_set = dataset.iloc[:cutoff_ind]
            # Criteria 1: Check if there's enough data left for a test set
            criteria_check[0] = cutoff_ind <= len(dataset)
            # Criteria 2: min number of events in test
            criteria_check[1] = test_set['sz_onset'].sum() >= self.n_min_events_test

            if not all(criteria_check):
                print(
                    f"Initial split: failed criteria {[i+1 for i, val in enumerate(criteria_check) if not val]} (trial {t+1})", end='\r')

                if not criteria_check[0]:
                    return dataset.iloc[cutoff_ind].name
                elif not criteria_check[1]:
                    cutoff_ind += 1

            t += 1

        return dataset.iloc[cutoff_ind].name

    def plot(self, dataset):
        ''' Plots the TSCV folds with the available data.

        Parameters
        ---------- 
        dataset : Dataset
            Instance of Dataset.
        '''
        if self.split_ind_ts is None:
            raise AttributeError(
                "Object has no attribute 'split_ind_ts'. Make sure the 'split' method has been run beforehand.")

        fig = go.Figure()

        for ifold in range(self.n_folds):

            train_set = dataset.metadata.loc[self.split_ind_ts[ifold, 0]: self.split_ind_ts[ifold, 1]]
            test_set = dataset.metadata.loc[self.split_ind_ts[ifold, 1]: self.split_ind_ts[ifold, 2]]

            # handle missing data between files
            train_set = self._handle_missing_data(train_set, ifold+1)
            test_set = self._handle_missing_data(test_set, ifold+1)

            # add existant data
            fig.add_trace(self._get_scatter_plot(train_set, color=COLOR_PALETTE[0]))
            fig.add_trace(self._get_scatter_plot(test_set, color=COLOR_PALETTE[1]))

            # add seizures
            fig.add_trace(self._get_scatter_plot_sz(
                train_set[train_set['sz_onset'] == 1],
                color=COLOR_PALETTE[0]
            ))
            fig.add_trace(self._get_scatter_plot_sz(
                test_set[test_set['sz_onset'] == 1],
                color=COLOR_PALETTE[1]
            ))

        # Config plot layout
        fig.update_yaxes(
            gridcolor='lightgrey',
            autorange='reversed',
            tickvals=list(range(1, self.n_folds+1)),
            ticktext=[f'Fold {i}  ' for i in range(1, self.n_folds+1)],
            tickfont=dict(size=12),
        )
        fig.update_layout(
            title='Time Series Cross Validation',
            showlegend=False,
            plot_bgcolor='white')
        fig.show()

    def _handle_missing_data(self, dataset, ind):
        """Internal method that updates the received dataset with NaN corresponding to where there are no files containing data."""

        # Create row for nan after files that are not contiguous (after file duration)
        missing_data = dataset[(dataset.index.diff()[1:] > dataset['total_duration']
                                [:-1]).tolist() + [False]].astype({'sz_onset': 'Int64'})
        missing_data.set_index(missing_data.index + missing_data['total_duration'], inplace=True)
        missing_data.iloc[:, :] = ['', 0, np.nan]

        dataset.insert(0, 'data', ind)
        missing_data.insert(0, 'data', np.nan)

        # Convert timestamp to datetime
        dataset.index = pd.to_datetime(dataset.index, unit='s', utc=True)
        missing_data.index = pd.to_datetime(missing_data.index, unit='s', utc=True)

        dataset = pd.concat([dataset, missing_data], ignore_index=False)
        dataset.sort_index(inplace=True)

        return dataset

    def _get_scatter_plot(self, dataset, color):
        """Internal method that returns a line-scatter-plot where data exists."""
        return go.Scatter(
            x=dataset.index,
            y=dataset.data,
            mode='lines',
            line={
                'color': 'rgba' + str(hex_to_rgba(
                    h=color,
                    alpha=1
                )),
                'width': 7
            }
        )

    def _get_scatter_plot_sz(self, dataset, color):
        """Internal method that returns a marker-scatter-plot where sz onsets exist."""
        return go.Scatter(
            x=dataset.index,
            y=dataset.data-0.1,
            mode='text',
            text=['ϟ'] * len(dataset),
            textfont=dict(
                size=16,
                color=color  # Set the color of the Unicode text here
            )
        )

    def iterate(self, h5dataset):
        ''' Iterates over the TSCV folds and at each iteration returns a train set and a test set. 

        Parameters
        ---------- 
        h5dataset : HDF5 file
            HDF5 file object with the following datasets:
            - "data": each entry corresponds to a sample with shape (embedding shape), e.g. (#features, ) or (sample duration, #channels) 
            - "timestamps": contains the start timestamp (unix in seconds) of each sample in the "data" dataset, with shape (#samples, ).
            - "annotations": contains the labels (0: interictal, 1: preictal) for each sample in the "data" dataset, with shape (#samples, ).
            - "sz_onsets": contains the Unix timestamps of the onsets of seizures (#sz_onsets, ). 

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

        for train_start_ts, test_start_ts, test_end_ts in self.split_ind_ts:

            train_indx = np.where(np.logical_and(timestamps >= train_start_ts, timestamps < test_start_ts))
            test_indx = np.where(np.logical_and(timestamps >= test_start_ts, timestamps < test_end_ts))

            train_sz_indx = np.where(np.logical_and(sz_onsets >= train_start_ts, sz_onsets < test_start_ts))
            test_sz_indx = np.where(np.logical_and(sz_onsets >= test_start_ts, sz_onsets < test_end_ts))

            yield (
                (h5dataset['data'][train_indx], h5dataset['annotations'][train_indx],
                 h5dataset['timestamps'][train_indx], sz_onsets[train_sz_indx]),
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

    def __init__(self, files_metadata, sz_onsets, sampling_frequency):
        self.files_metadata = files_metadata.astype({'first_timestamp': 'int64', 'total_duration': 'int64'})
        self.sz_onsets = np.array(sz_onsets)
        self.sampling_frequency = sampling_frequency

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
