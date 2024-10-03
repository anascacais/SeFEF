"""
Module Name: evaluation
Description: Contains functions to impement time-series cross validation (TSCV)
Author: Ana Sofia Carmo
Date: Oct 1, 2024
Version: 0.0.1
License: MIT License
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
    n_min_events : int, optional
        Minimum number of lead seizures to include in the train set. Should guarantee at least one lead seizure is left for testing. Defaults to 3.
    initial_train_duration : int, optional
        Set duration of train for initial split (in seconds). Defaults to 1/3 of total recorded duration.
    test_duration : int, optional
        Set duration of test (in seconds). Defaults to 1/2 of 'initial_train_duration'.
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
    
    Raises
    -------
    ValueError :
        Raised whenever TSCV is not passible to be performed under the attributes set by the user and available data. 
    AttributeError :
        Raised when 'plot' is called before 'split'.
    ''' 

    def __init__(self, n_min_events=3, initial_train_duration=None, test_duration=None):
        self.n_min_events = n_min_events
        self.initial_train_duration = initial_train_duration
        self.test_duration = test_duration
        self.method = 'expanding'

        self.n_folds = None
        self.split_ind_ts = None

    def split(self, dataset, iteratively=True):
        """ Get timestamp indices to split data for time series cross-validation. 
        - The train set would be given by metadata.loc[train_start_ts : test_start_ts].
        - The test set would be given by metadata.loc[test_start_ts : test_end_ts].
        
        Parameters:
        -----------
        dataset : Dataset
            Instance of Dataset.
        iteratively : bool, ooptional
            If the split is meant to return the timestamp indices for each fold iteratively (True) or to simply update 'split_ind_ts' (False). Defaults to True.

        Returns:
        -------
        train_start_ts : int
            Timestamp index for the start of the train set.
        test_start_ts : int
            Timestamp index for the start of the test set (and end of train set).
        test_end_ts : int
            Timestamp index for the end of the test set.
        """
        if self.initial_train_duration is None:
            total_recorded_duration = dataset.files_metadata['total_duration'].sum()
            self.initial_train_duration = (1/3) * total_recorded_duration

        if self.test_duration is None:
            self.test_duration = (1/2) * self.initial_train_duration

        # Check basic conditions
        if dataset.metadata['sz_onset'].sum() < self.n_min_events + 1:
            raise ValueError("Dataset does not contain the minimum number of events. Just give up (or change the value of 'n_min_events').")

        if dataset.files_metadata['total_duration'].sum() < self.initial_train_duration + self.test_duration:
            raise ValueError("Dataset does not contain enough data to do this split. Just give up (or decrease 'initial_train_duration' and/or 'test_duration').")

        # Get index for initial split
        initial_cutoff_ts = self._get_cutoff_ts(dataset)
        initial_cutoff_ts = self._check_criteria(dataset, initial_cutoff_ts)

        if iteratively:
            return self._expanding_window_split(dataset, initial_cutoff_ts)
        else: 
            for _ in self._expanding_window_split(dataset, initial_cutoff_ts):
                pass
            return None



    def _expanding_window_split(self, dataset, initial_cutoff_ts):
        """Internal method for expanding window cross-validation."""

        after_train_set = dataset.metadata.loc[initial_cutoff_ts:]
        self.n_folds = int(after_train_set['total_duration'].sum() // self.test_duration)
        self.split_ind_ts = np.zeros((self.n_folds, 3), dtype='int64')
        
        train_start_ts = dataset.metadata.index[0]
        test_start_ts = initial_cutoff_ts.copy()
        
        for i in range(self.n_folds):
            if i != 0:
                after_train_set = dataset.metadata.loc[test_end_ts:]
                test_start_ts = test_end_ts

            test_end_ts = after_train_set.index[after_train_set['total_duration'].cumsum() > self.test_duration].tolist()[0]
            self.split_ind_ts[i, :] = [train_start_ts, test_start_ts, test_end_ts]
            
            yield train_start_ts, test_start_ts, test_end_ts
                

    def _sliding_window_split(self):
        """Internal method for sliding window cross-validation."""
        pass

    def _get_cutoff_ts(self, dataset):
        """Internal method for getting the first iteration of the cutoff timestamp based on 'self.initial_train_duration'."""
        cutoff_ts = dataset.metadata.index[dataset.metadata['total_duration'].cumsum() > self.initial_train_duration].tolist()[0]
        return cutoff_ts


    def _check_criteria(self, dataset, initial_cutoff_ts):
        """Internal method for iterating the initial cutoff timestamp in order to respect the condition on the minimum number of seizures."""

        criteria_check = [False] * 2

        initial_cutoff_ind = dataset.metadata.index.get_loc(initial_cutoff_ts)

        while not all(criteria_check):
            initial_train_set = dataset.metadata.iloc[:initial_cutoff_ind]
            after_train_set = dataset.metadata.iloc[initial_cutoff_ind:]
            
            # Criteria 1: min number of events in train
            criteria_check[0] = initial_train_set['sz_onset'].sum() >= self.n_min_events
            # Criteria 2: min number of events in test
            criteria_check[1] = after_train_set['sz_onset'].sum() >= 1
            
            if not all(criteria_check):
                print(f"Failed criteria {[i+1 for i, val in enumerate(criteria_check) if not val]}")
                
                if (not criteria_check[0]) and (not criteria_check[1]):
                    raise ValueError("Dataset does not comply with the conditions for this split. Just give up (or decrease 'n_min_events', 'initial_train_duration', and/or 'test_duration').")
                elif not criteria_check[0]:
                    initial_cutoff_ind += 1
                elif not criteria_check[1]:
                    initial_cutoff_ind -= 1
        
        # Check if there's enough data left for at least one test set
        if after_train_set['total_duration'].sum() < self.test_duration:
            raise ValueError("Dataset does not comply with the conditions for this split. Just give up (or decrease 'n_min_events', 'initial_train_duration', and/or 'test_duration').")

        return dataset.metadata.iloc[initial_cutoff_ind].name

    def plot(self, dataset):
        ''' Plots the TSCV folds with the available data.
        
        Parameters
        ---------- 
        dataset : Dataset
            Instance of Dataset.
        ''' 
        if self.split_ind_ts is None:
            raise AttributeError("Object has no attribute 'split_ind_ts'. Make sure the 'split' method has been run beforehand.")
        
        fig = go.Figure()
        
        for ifold in range(self.n_folds):

            train_set = dataset.metadata.loc[self.split_ind_ts[ifold, 0] : self.split_ind_ts[ifold, 1]]
            test_set = dataset.metadata.loc[self.split_ind_ts[ifold, 1] : self.split_ind_ts[ifold, 2]]

            # handle missing data between files
            train_set = self._handle_missing_data(train_set, ifold+1)
            test_set = self._handle_missing_data(test_set, ifold+1)

            # add existant data 
            fig.add_trace(go.Scatter(
                x = train_set.index, 
                y = train_set.data, 
                mode = 'lines',
                line = {
                    'color': 'rgba' + str(hex_to_rgba(
                        h = COLOR_PALETTE[0],
                        alpha=1
                    )),
                    'width': 7
                }
                )
            )
            
            fig.add_trace(go.Scatter(
                x = test_set.index, 
                y = test_set.data, 
                mode = 'lines',
                line = {
                    'color': 'rgba' + str(hex_to_rgba(
                        h = COLOR_PALETTE[1],
                        alpha=1
                    )),
                    'width': 7
                }
                )
            )

        # Config plot layout
        fig.update_yaxes(
            gridcolor = 'lightgrey',
            autorange = 'reversed',
            tickvals = list(range(1, self.n_folds+1)),
            ticktext = [f'Fold {i}  ' for i in range(1, self.n_folds+1)],
            tickfont = dict(size=12),
        )
        fig.update_layout(
            title = 'Time Series Cross Validation',
            showlegend = False,
            plot_bgcolor = 'white')
        fig.show()

    def _handle_missing_data(self, dataset, ind):

        # Create row for nan after files that are not contiguous (after file duration)
        missing_data = dataset[(dataset.index.diff()[1:] > dataset['total_duration'][:-1]).tolist() + [False]]
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


        



class Dataset:
    ''' Create a Dataset with metadata on the data that will be used for training and testing
    
    Attributes
    ---------- 
    files_metadata: pd.DataFrame
        Input DataFrame with the following columns:
        - 'filepath' (str): Path to each file containing data.
        - 'first_timestamp' (int64): The Unix-time timestamp (in seconds) of the first sample of each file.
        - 'total_duration' (int): Total duration of file in seconds (equivalent to #samples * sampling_frequency)
        It is expected that data within each file is non-overlapping in time and that there are no time gaps between samples in the file. 
    sz_onsets: array-like
        Contains the Unix-time timestamps (in seconds) corresponding to the onsets of seizures.
    sampling_frequency: int
        Frequency at which the data is stored in each file.
    ''' 

    def __init__(self, files_metadata, sz_onsets, sampling_frequency):
        self.files_metadata = files_metadata
        self.sz_onsets = sz_onsets
        self.sampling_frequency = sampling_frequency

        self.metadata = self._get_metadata()

    def _get_metadata(self):
        """Internal method that updates 'self.metadata' by placing each seizure onset within an acquisition file."""
        files_metadata = self.files_metadata.copy()
        files_metadata.set_index(pd.Index(files_metadata['first_timestamp'], dtype='int64'), inplace=True)
        files_metadata.drop(['first_timestamp'], axis=1, inplace=True) 

        sz_onsets = pd.DataFrame([1]*len(self.sz_onsets), index=pd.Index(self.sz_onsets, dtype='int64'), columns=['sz_onset'])
        
        return files_metadata.join(sz_onsets, how='outer')
        
        


    
