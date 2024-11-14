import unittest
import pandas as pd
import numpy as np

from sefef.evaluation import TimeSeriesCV, Dataset

class TestTimeSeriesCV(unittest.TestCase):

    def setUp(self):
        self.files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800, 1609460100, 1609460400, 1609460700],
            'total_duration': [300, 300, 300, 300, 300, 300]  # 5 minutes per file
        })
        self.sz_onsets = [1609459500, 1609459800, 1609460700]
        self.sampling_frequency = 128  #Hz
        self.dataset = Dataset(self.files_metadata, self.sz_onsets, self.sampling_frequency)
        self.tscv = TimeSeriesCV(n_min_events_train=1, n_min_events_test=1)
    
    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.tscv.n_min_events_train, int))
        self.assertTrue(isinstance(self.tscv.n_min_events_test, int))
        self.assertTrue(isinstance(self.tscv.initial_train_duration, (int, type(None))))
        self.assertTrue(isinstance(self.tscv.test_duration, (int, type(None))))
        self.assertTrue(isinstance(self.tscv.method, str))
        self.assertTrue(isinstance(self.tscv.n_folds, type(None)))
        self.assertTrue(isinstance(self.tscv.split_ind_ts, type(None)))

    # 2. Split when all is standard
    def test_split(self):
        expected_split_ind_ts = np.array([
            [1609459200, 1609459800, 1609460100],
            [1609459200, 1609460100, 1609461000]
        ])
        expected_n_folds = 2
        self.tscv.split(self.dataset, iteratively=False, plot=False)
        
        self.assertTrue(self.tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(self.tscv.split_ind_ts, expected_split_ind_ts)

    # 3. Split when Dataset instance is empty
    def test_split_empty_dataset(self):
        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, [], self.sampling_frequency)

        with self.assertRaises(ValueError):
            self.tscv.split(dataset, iteratively=False, plot=False)

    # 4. Split when total duration of Dataset is smaller than initial_train_duration + test_duration
    def test_split_small_dataset(self):
        tscv = TimeSeriesCV(n_min_events_train=1, n_min_events_test=1, initial_train_duration=1600, test_duration=300)

        with self.assertRaises(ValueError):
            tscv.split(self.dataset, iteratively=False, plot=False)

    # 5. Split when number of events in Dataset is smaller than n_min_events_train + n_min_events_test
    def test_split_no_events(self):
        tscv = TimeSeriesCV(n_min_events_train=3, n_min_events_test=1)

        with self.assertRaises(ValueError):
            tscv.split(self.dataset, iteratively=False, plot=False)

    # 6. Split when all events are within initial_train_duration
    def test_split_events_early(self):
        expected_split_ind_ts = np.array([
            [1609459200, 1609459500, 1609459800],
        ])
        expected_n_folds = 1

        sz_onsets = [1609459200, 1609459500]
        dataset = Dataset(self.files_metadata, sz_onsets, self.sampling_frequency)
        self.tscv.split(dataset, iteratively=False, plot=False)

        self.assertTrue(self.tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(self.tscv.split_ind_ts, expected_split_ind_ts)


    # 7. Split when all events are after initial_train_duration
    def test_split_events_late(self):
        expected_split_ind_ts = np.array([
            [1609459200, 1609460700, 1609461000],
        ])
        expected_n_folds = 1

        sz_onsets = [1609460400, 1609460700]
        dataset = Dataset(self.files_metadata, sz_onsets, self.sampling_frequency)
        self.tscv.split(dataset, iteratively=False, plot=False)

        self.assertTrue(self.tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(self.tscv.split_ind_ts, expected_split_ind_ts)





class TestDataset(unittest.TestCase):
    def setUp(self):
        self.files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800],
            'total_duration': [300, 300, 300]  # 5 minutes per file
        })
        self.sz_onsets = [1609459520]
        self.sampling_frequency = 128  #Hz
        self.dataset = Dataset(self.files_metadata, self.sz_onsets, self.sampling_frequency)

    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.dataset.files_metadata, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.sz_onsets, (np.ndarray)))
        self.assertTrue(isinstance(self.dataset.sampling_frequency, int))
        self.assertTrue(isinstance(self.dataset.metadata, pd.DataFrame))

    # 2. Test Metadata Calculation:  Ensure that _get_metadata places seizure onsets in the correct files.
    def test_get_metadata(self):
        expected_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', np.nan],
            'total_duration': [300, 300, 300, 0],
            'sz_onset': [0, 1, 0, 0],
        }, index=pd.Series([1609459200, 1609459500, 1609459800, 1609460100], dtype='int64'))
        expected_metadata = expected_metadata.astype({'filepath': str, 'sz_onset': 'int64', 'total_duration': 'int64'})

        pd.testing.assert_frame_equal(self.dataset.metadata, expected_metadata)

    # 3.a) Test Edge Cases: No Seizure Onsets Ensure the method works if sz_onsets is empty.
    def test_no_sz_onsets(self):
        dataset = Dataset(self.files_metadata, [], self.sampling_frequency)
        self.assertTrue(dataset.metadata['sz_onset'].sum()==0)

    # 3.b) Test Edge Cases: Empty Metadata Test behavior when files_metadata is empty.
    def test_empty_files_with_onset_metadata(self):
        expected_metadata = pd.DataFrame({
            'filepath': [np.nan, np.nan],
            'total_duration': [0, 0],
            'sz_onset': [1, 0]
        }, index=pd.Series([1609459520, 1609459520], dtype='int64'))
        expected_metadata = expected_metadata.astype({'filepath': str, 'sz_onset': 'int64', 'total_duration': 'int64'})

        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, self.sz_onsets, self.sampling_frequency)

        pd.testing.assert_frame_equal(dataset.metadata, expected_metadata)

    # 3.c) Test Edge Cases: Empty Metadata Test behavior when files_metadata is empty.
    def test_empty_files_metadata(self):
        expected_metadata = pd.DataFrame(columns=['filepath', 'total_duration', 'sz_onset'], index=pd.Series([], dtype='int64'))
        expected_metadata = expected_metadata.astype({'filepath': str, 'sz_onset': 'int64', 'total_duration': 'int64'})

        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, [], self.sampling_frequency)

        pd.testing.assert_frame_equal(dataset.metadata, expected_metadata)

    # 3.d) Test Edge Cases: Mismatched Time Periods Ensure the method handles onsets outside the range of files_metadata.
    def test_out_of_range_onsets(self):
        out_of_range_onsets = [1609459100, 1609459900]  # Before and within file ranges

        expected_metadata = pd.DataFrame({
            'filepath': [np.nan, 'file1.csv', 'file2.csv', 'file3.csv', np.nan],
            'total_duration': [0, 300, 300, 300, 0],
            'sz_onset': [1, 0, 0, 1, 0]
        }, index=pd.Series([1609459100, 1609459200, 1609459500, 1609459800, 1609460100], dtype='int64'))
        expected_metadata = expected_metadata.astype({'filepath': str, 'sz_onset': 'int64', 'total_duration': 'int64'})

        dataset = Dataset(self.files_metadata, out_of_range_onsets, self.sampling_frequency)
        
        pd.testing.assert_frame_equal(dataset.metadata, expected_metadata)


if __name__ == '__main__':
    unittest.main()
