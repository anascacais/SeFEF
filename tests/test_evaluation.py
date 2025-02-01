import unittest
import pandas as pd
import numpy as np
import h5py
from unittest.mock import MagicMock, patch

from sefef.evaluation import TimeSeriesCV, Dataset

class TestTimeSeriesCV(unittest.TestCase):

    def setUp(self):
        self.files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv', 'file7.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800, 1609460100, 1609460400, 1609460700, 1609461000],
            'total_duration': [300, 300, 300, 300, 300, 300, 300]  # 5 minutes per file
        })
        self.sz_onsets = [1609459800, 1609461000]
        self.preictal_duration = 300
        self.prediction_latency = 300
        self.pre_lead_sz_interval = 900
        self.post_sz_interval = 300
        self.dataset = Dataset(self.files_metadata, self.sz_onsets)
        self.tscv = TimeSeriesCV(preictal_duration=self.preictal_duration, prediction_latency=self.prediction_latency, n_min_events_train=1, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)
    
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
            [1609459200, 1609460400, 1609461300],
        ])
        expected_n_folds = 1
        self.tscv.split(self.dataset, iteratively=False, plot=False)
        
        self.assertTrue(self.tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(self.tscv.split_ind_ts, expected_split_ind_ts)

    # 3. Split when Dataset instance is empty
    def test_split_empty_dataset(self):
        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, [])

        with self.assertRaises(ValueError):
            self.tscv.split(dataset, iteratively=False, plot=False)

    # 4. Split when total duration of Dataset is smaller than initial_train_duration + test_duration
    def test_split_small_dataset(self):
        files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800, 1609460100, 1609460400, 1609460700],
            'total_duration': [300, 300, 300, 300, 300, 300]  # 5 minutes per file
        })
        sz_onsets = [1609459800, 1609460700]
        self.dataset = Dataset(files_metadata, sz_onsets)
        tscv = TimeSeriesCV(preictal_duration=self.preictal_duration, prediction_latency=self.prediction_latency, n_min_events_train=1, n_min_events_test=1, initial_train_duration=1600, test_duration=300, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)

        with self.assertRaises(ValueError):
            tscv.split(self.dataset, iteratively=False, plot=False)

    # 5. Split when number of events in Dataset is smaller than n_min_events_train + n_min_events_test
    def test_split_no_events(self):
        tscv = TimeSeriesCV(preictal_duration=self.preictal_duration, prediction_latency=self.prediction_latency, n_min_events_train=3, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)

        with self.assertRaises(ValueError):
            tscv.split(self.dataset, iteratively=False, plot=False)

    # 6. Split when there are seizure onsets but not enough preictal data 
    def test_split_not_enough_preictal(self):
        files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv'],
            'first_timestamp': [1609459500, 1609459800, 1609460100, 1609460400, 1609460700],
            'total_duration': [300, 300, 300, 300, 300]  # 5 minutes per file
        })
        sz_onsets = [1609459800, 1609460700]
        dataset = Dataset(files_metadata, sz_onsets)
        with self.assertRaises(ValueError):
            self.tscv.split(dataset, iteratively=False, plot=False)

    # 7. Case equal to previous but there is no prediction latency (edge case)
    def test_split_barely_enough_preictal(self):
        files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv'],
            'first_timestamp': [1609459500, 1609459800, 1609460100, 1609460400, 1609460700],
            'total_duration': [300, 300, 300, 300, 300]  # 5 minutes per file
        })
        sz_onsets = [1609459800, 1609460700]
        expected_n_folds = 1
        expected_split_ind_ts = np.array([
            [1609459500, 1609460400, 1609461000],
        ])

        dataset = Dataset(files_metadata, sz_onsets)
        tscv = TimeSeriesCV(preictal_duration=self.preictal_duration, prediction_latency=0, n_min_events_train=1, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)
        tscv.split(dataset, iteratively=False, plot=False)

        self.assertTrue(tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(tscv.split_ind_ts, expected_split_ind_ts)

    # 8. Split accounting for non-lead seizures
    def test_effect_non_lead(self):
        files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv', 'file7.csv', 'file8.csv', 'file9.csv', 'file6.csv', 'file7.csv'],
            'first_timestamp': np.arange(1609459500, 1609462800, 300).tolist(),
            'total_duration': [300] * 11  # 5 minutes per file
        })
        sz_onsets = [1609460100, 1609460700, 1609462500]
        expected_n_folds = 1
        expected_split_ind_ts = np.array([
            [1609459500, 1609460700, 1609462800],
        ])
        dataset = Dataset(files_metadata, sz_onsets)
        tscv = TimeSeriesCV(preictal_duration=self.preictal_duration, prediction_latency=0, n_min_events_train=1, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)
        tscv.split(dataset, iteratively=False, plot=False)

        self.assertTrue(tscv.n_folds == expected_n_folds)
        np.testing.assert_array_equal(tscv.split_ind_ts, expected_split_ind_ts)


    # 9. Iterate removing non-lead seizures from train
    @patch("h5py.File", autospec=True)
    def test_iterate_non_lead(self, mock_h5py_file):
        files_metadata = pd.DataFrame({
            'filepath': [f'file{i}.csv' for i in range(1, 14)],
            'first_timestamp': np.arange(1609459500, 1609459500+300*13, 300).tolist(),
            'total_duration': [300] * 13  # 5 minutes per file
        })
        preictal_duration = 600
        sz_onsets = [1609460100, 1609460400, 1609461900, 1609463100]

        dataset = Dataset(files_metadata, sz_onsets)
        tscv = TimeSeriesCV(preictal_duration=preictal_duration, prediction_latency=self.prediction_latency, n_min_events_train=2, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)
        tscv.split(dataset, iteratively=False, plot=False)

        # Mock HDF5 file behavior
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = ['data', 'timestamps', 'annotations', 'sz_onsets']
        mock_file.__getitem__.side_effect = {
            'data': np.array([None]*13),
            'timestamps': np.arange(1609459500, 1609459500+300*13, 300),
            'annotations': np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
            'sz_onsets': np.array(sz_onsets),
        }.__getitem__
        mock_h5py_file.return_value = mock_file
        
        with h5py.File('test_file.h5', 'r+') as h5dataset:
            iterator = tscv.iterate(h5dataset)
        expected = (
            (
                np.array([None, None, None]),
                np.array([1, 1, 1]),
                np.array([1609459500, 1609461000, 1609461300]),
                np.array([1609460100, 1609461900])
            ),
            (
                np.array([None, None, None]),
                np.array([1, 0, 0]),
                np.array([1609462500, 1609462800, 1609463100]),
                np.array([1609463100])
            )
        )

        output_iterator = list(iterator)[0]

        self.assertTrue(len(expected) == len(output_iterator))
        for expected_tuple, output_tuple in zip(expected, output_iterator):
            for exp_arr, out_arr in zip(expected_tuple, output_tuple):
                np.testing.assert_array_equal(out_arr, exp_arr)

    # 10. Same as previous but for analogous function
    @patch("h5py.File", autospec=True)
    def test_iterate_non_lead(self, mock_h5py_file):
        files_metadata = pd.DataFrame({
            'filepath': [f'file{i}.csv' for i in range(1, 14)],
            'first_timestamp': np.arange(1609459500, 1609459500+300*13, 300).tolist(),
            'total_duration': [300] * 13  # 5 minutes per file
        })
        preictal_duration = 600
        sz_onsets = [1609460100, 1609460400, 1609461900, 1609463100]

        dataset = Dataset(files_metadata, sz_onsets)
        tscv = TimeSeriesCV(preictal_duration=preictal_duration, prediction_latency=self.prediction_latency, n_min_events_train=2, n_min_events_test=1, pre_lead_sz_interval=self.pre_lead_sz_interval, post_sz_interval=self.post_sz_interval)
        tscv.split(dataset, iteratively=False, plot=False)

        # Mock HDF5 file behavior
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = ['data', 'timestamps', 'annotations', 'sz_onsets']
        mock_file.__getitem__.side_effect = {
            'data': np.array([None]*13),
            'timestamps': np.arange(1609459500, 1609459500+300*13, 300),
            'annotations': np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]),
            'sz_onsets': np.array(sz_onsets),
        }.__getitem__
        mock_h5py_file.return_value = mock_file
        
        with h5py.File('test_file.h5', 'r+') as h5dataset:
            output_tuple = tscv.get_TSCV_fold(h5dataset, 0)

        expected_tuple = (
            (
                np.array([None, None, None]),
                np.array([1, 1, 1]),
                np.array([1609459500, 1609461000, 1609461300]),
                np.array([1609460100, 1609461900])
            ),
            (
                np.array([None, None, None]),
                np.array([1, 0, 0]),
                np.array([1609462500, 1609462800, 1609463100]),
                np.array([1609463100])
            )
        )

        for exp_arr, out_arr in zip(expected_tuple[0], output_tuple[0]):
            np.testing.assert_array_equal(out_arr, exp_arr)
        for exp_arr, out_arr in zip(expected_tuple[1], output_tuple[1]):
            np.testing.assert_array_equal(out_arr, exp_arr)



class TestDataset(unittest.TestCase):
    def setUp(self):
        self.files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800],
            'total_duration': [300, 300, 300]  # 5 minutes per file
        })
        self.sz_onsets = [1609459520]
        self.dataset = Dataset(self.files_metadata, self.sz_onsets)

    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.dataset.files_metadata, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.sz_onsets, (np.ndarray)))
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
        dataset = Dataset(self.files_metadata, [])
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
        dataset = Dataset(empty_files_metadata, self.sz_onsets)

        pd.testing.assert_frame_equal(dataset.metadata, expected_metadata)

    # 3.c) Test Edge Cases: Empty Metadata Test behavior when files_metadata is empty.
    def test_empty_files_metadata(self):
        expected_metadata = pd.DataFrame(columns=['filepath', 'total_duration', 'sz_onset'], index=pd.Series([], dtype='int64'))
        expected_metadata = expected_metadata.astype({'filepath': str, 'sz_onset': 'int64', 'total_duration': 'int64'})

        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, [])

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

        dataset = Dataset(self.files_metadata, out_of_range_onsets)
        
        pd.testing.assert_frame_equal(dataset.metadata, expected_metadata)


if __name__ == '__main__':
    unittest.main()