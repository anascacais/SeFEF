"""
Module Name: evaluation
Description: Contains functions to impement time-series cross validation (TSCV)
Author: Ana Sofia Carmo
Date: Oct 1, 2024
Version: 0.0.1
License: MIT License
"""

# third-party
import numpy as np
import pandas as pd


class TimeSeriesCV:
    ''' Class description 
    
    Attributes
    ---------- 
    n_min_events: int, optional
        Minimum number of lead seizures to include in the train set. Defaults to 3.
    initial_train_duration: int, optional
        Set duration of train for initial split. Defaults to 1/3 of total recorded duration.
    test_duration: int, optional
        Set duration of test. Defaults to 1/2 of 'initial_train_duration'.
    method: str
        Description
    n_folds: int
        Description.
    
    Methods
    -------
    split: 
        Description
    
    Raises
    -------
    ValueError:
        Description
    ''' 

    def __init__(self, n_min_events=3, initial_train_duration=None, test_duration=None):
        self.n_min_events = n_min_events
        self.initial_train_duration = initial_train_duration
        self.test_duration = test_duration
        self.method = 'expanding'

        self.n_folds = None

    def split(self, dataset):
        """
        Get index to split data for time series cross-validation.
        
        Parameters:
        -----------
        dataset : Dataset
            Instance of Dataset.
        Returns:
        -------
        train_idx : array-like, shape (n_train,)
            The indices for the training set.
        test_idx : array-like, shape (n_test,)
            The indices for the test set.
        """
        if self.initial_train_duration is None:
            total_recorded_duration = dataset.files_metadata['total_duration'].sum()
            self.initial_train_duration = (1/3) * total_recorded_duration

        if self.test_duration is None:
            self.test_duration = (1/2) * self.initial_train_duration

        # Check basic conditions
        if dataset.metadata['sz_onset'].sum() < self.n_min_events:
            raise ValueError("Dataset does not contain the minimum number of events. Just give up (or change the value of 'n_min_events').")

        if dataset.files_metadata['total_duration'].sum() < self.initial_train_duration + self.test_duration:
            raise ValueError("Dataset does not contain enough data to do this split. Just give up (or decrease 'initial_train_duration' and/or 'test_duration').")

        # Get index for initial split
        initial_cutoff_ts = self._get_cutoff_ts(dataset)
        initial_cutoff_ts = self._check_criteria(dataset, initial_cutoff_ts)

        return self._expanding_window_split(dataset, initial_cutoff_ts)



    def _expanding_window_split(self, dataset, initial_cutoff_ts):
        """Internal method for expanding window cross-validation."""

        after_train_set = dataset.metadata.loc[initial_cutoff_ts:]
        self.n_folds = int(after_train_set['total_duration'].sum() // self.test_duration)
        
        cutoff_ts = initial_cutoff_ts.copy()
        
        for i in range(self.n_folds):
            if i != 0:
                after_train_set = dataset.metadata.loc[cutoff_ts:]
                cutoff_ts = after_train_set.index[after_train_set['total_duration'].cumsum() > self.test_duration].tolist()[0]
            yield cutoff_ts
                

    def _sliding_window_split(self):
        """Internal method for sliding window cross-validation."""
        pass

    def _get_cutoff_ts(self, dataset):
        """Internal method for getting the first iteration of the cutoff timestamp based on 'self.initial_train_duration'."""
        cutoff_ts = dataset.metadata.index[dataset.metadata['total_duration'].cumsum() > self.initial_train_duration].tolist()[0]
        return cutoff_ts


    def _check_criteria(self, dataset, initial_cutoff_ts):
        """Internal method for iterating the initial cutoff timestamp in order to respect the condition on the minimum number of seizures."""

        criteria_check = [False] * 1

        initial_cutoff_ind = dataset.metadata.index.get_loc(initial_cutoff_ts)

        while not all(criteria_check):
            initial_train_set = dataset.metadata.iloc[:initial_cutoff_ind]
            
            # Criteria 1: min number 
            criteria_check[0] = initial_train_set['sz_onset'].sum() >= self.n_min_events
            
            if not all(criteria_check):
                print(f"Failed criteria {[i+1 for i, val in enumerate(criteria_check) if not val]}")
                
                if not criteria_check[0]:
                    initial_cutoff_ind += 1
        
        # Check if there's enough data left for at least one test set
        after_train_set = dataset.metadata.iloc[initial_cutoff_ind:]
        if after_train_set['total_duration'].sum() < self.test_duration:
            raise ValueError("Dataset does not comply with the conditions for this split. Just give up (or decrease 'n_min_events', 'initial_train_duration', and/or 'test_duration').")

        return dataset.metadata.iloc[initial_cutoff_ind].name




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
        
        


    
