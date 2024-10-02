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
        description
    
    Methods
    -------
    split: 
        description
    
    Raises
    -------
    ValueError:
        description
    ''' 

    def __init__(self, n_min_events=3, initial_train_duration=None, test_duration=None):
        self.n_min_events = n_min_events
        if initial_train_duration is not None:
            self.initial_train_duration = initial_train_duration
        else: 
            self.initial_train_duration = None
        self.test_duration = test_duration

        self.method = 'expanding'

    def split(self, dataset):
        """
        Get index to split data for time series cross-validation.
        
        Parameters:
        -----------
        dataset : Dataset
            
        Returns:
        -------
        train_idx : array-like, shape (n_train,)
            The indices for the training set.
        test_idx : array-like, shape (n_test,)
            The indices for the test set.
        """
        #if self.initial_train_duration is None: 
        init_cutoff_ind = self._get_cutoff_ind(dataset)

        return None

    def _expanding_window_split(self, n_samples):
        """Internal method for expanding window cross-validation."""
        fold_size = (n_samples - self.test_size) // self.n_splits
        for i in range(self.n_splits):
            train_idx = np.arange(0, fold_size * (i + 1))
            test_idx = np.arange(fold_size * (i + 1), fold_size * (i + 1) + self.test_size)
            yield train_idx, test_idx

    def _sliding_window_split(self):
        """Internal method for sliding window cross-validation."""
        pass

    def _get_cutoff_ind(self, dataset):
        """Internal method for ????"""
        if self.initial_train_duration is None:
            total_recorded_duration = dataset.files_metadata['total_duration'].sum()
            self.initial_train_duration = (1/3) * total_recorded_duration

        # cutoff_ind = df_epochs.index[df_epochs["recorded time (min)"].cumsum() > (
        # total_recorded_time * (1/3))].tolist()[0]
        pass




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

        sz_onsets = pd.DataFrame([1]*len(self.sz_onsets), index=pd.Index(self.sz_onsets, dtype='int64'), columns=['sz_onset'])
        self.metadata = files_metadata.join(sz_onsets, how='outer')
        
        


    
