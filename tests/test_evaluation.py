import unittest
import pandas as pd
import numpy as np

from sefef.evaluation import TimeSeriesCV, Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.files_metadata = pd.DataFrame({
            'filepath': ['file1.csv', 'file2.csv', 'file3.csv'],
            'first_timestamp': [1609459200, 1609459500, 1609459800],
            'total_duration': [300, 300, 300]  # 5 minutes per file
        })
        self.sz_onsets = [1609459520]
        self.sampling_frequency = 1  # 1 Hz
        self.dataset = Dataset(self.files_metadata, self.sz_onsets, self.sampling_frequency)

    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.dataset.files_metadata, pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.sz_onsets, (list, np.ndarray)))
        self.assertTrue(isinstance(self.dataset.sampling_frequency, int))
        self.assertTrue(isinstance(self.dataset.metadata, pd.DataFrame))

    # 2. Test Metadata Calculation:  Ensure that _get_metadata places seizure onsets in the correct files.
    def test_get_metadata(self):
        expected_metadata = pd.DataFrame({
            'filepath': pd.Series(['file1.csv', 'file2.csv', 'file3.csv'], dtype=str),
            'total_duration': pd.Series([300, 300, 300], dtype='Int64'),
            'sz_onset': pd.Series([np.nan, 1, np.nan], dtype='Int64'),
        }, index=[1609459200, 1609459500, 1609459800])

        pd.testing.assert_frame_equal(self.dataset.metadata.fillna(0), expected_metadata.fillna(0))

    # 3.a) Test Edge Cases: No Seizure Onsets Ensure the method works if sz_onsets is empty.
    def test_no_sz_onsets(self):
        dataset = Dataset(self.files_metadata, [], self.sampling_frequency)
        self.assertTrue(dataset.metadata['sz_onset'].isna().all())

    # 3.b) Test Edge Cases: Empty Metadata Test behavior when files_metadata is empty.
    def test_empty_files_metadata(self):
        expected_metadata = pd.DataFrame({
            'filepath': pd.Series([np.nan], dtype=str),
            'total_duration': pd.Series([np.nan], dtype='Int64'),
            'sz_onset': pd.Series([1], dtype='Int64'),
        }, index=[1609459520])

        empty_files_metadata = pd.DataFrame(columns=['filepath', 'first_timestamp', 'total_duration'])
        dataset = Dataset(empty_files_metadata, self.sz_onsets, self.sampling_frequency)

        pd.testing.assert_frame_equal(dataset.metadata.fillna(0), expected_metadata.fillna(0))

    # # 3.c) Test Edge Cases: Mismatched Time Periods Ensure the method handles onsets outside the range of files_metadata.
    # def test_out_of_range_onsets(self):
    #     out_of_range_onsets = [1609459100, 1609459900]  # Before and after file ranges
    #     dataset = Dataset(self.files_metadata, out_of_range_onsets, self.sampling_frequency)
    #     self.assertTrue(dataset.metadata['sz_onset'].isna().all())


if __name__ == '__main__':
    unittest.main()
