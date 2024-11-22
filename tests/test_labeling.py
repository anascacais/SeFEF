import unittest
from unittest.mock import MagicMock, patch
import h5py
import numpy as np

from sefef.labeling import add_annotations, add_sz_onsets

class TestLabeling(unittest.TestCase):

    def setUp(self):
        self.timestamps = [1609459200, 1609459201, 1609459202, 1609459203, 1609459204, 1609459205, 1609459206, 1609459207, 1609459208, 1609459209, 1609459210]
        self.sz_onsets_ts = [1609459208]
        self.preictal_duration = 5 # seconds
        self.prediction_latency = 1 # seconds

    @patch("h5py.File", autospec=True)
    def test_add_annotations_to_empty_dataset(self, mock_h5py_file):
        # Mock HDF5 file behavior
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = []  # Simulate no datasets exist
        mock_h5py_file.return_value = mock_file

        with h5py.File('test_file.h5', 'r+') as h5dataset:
            with self.assertRaises(KeyError):
                add_annotations(h5dataset, self.sz_onsets_ts, self.preictal_duration, self.prediction_latency)
    
    @patch("h5py.File", autospec=True)
    def test_add_dataset_success(self, mock_h5py_file):
        # Mock HDF5 file behavior
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = ['timestamps']
        mock_file.__getitem__.side_effect = {'timestamps': np.array(self.timestamps)}.__getitem__
        mock_h5py_file.return_value = mock_file
        
        with h5py.File('test_file.h5', 'r+') as h5dataset:
            add_annotations(h5dataset, self.sz_onsets_ts, self.preictal_duration, self.prediction_latency)

        expected_labeling = [False, False, True, True, True, True, True, False, False, False, False]
        
        # Check if the dataset was correctly
        args, kwargs = mock_file.create_dataset.call_args
        self.assertEqual(args[0], 'annotations')  # Check dataset name
        np.testing.assert_array_equal(kwargs['data'], expected_labeling)   # Check data content

    @patch("h5py.File", autospec=True)
    def test_labeling_onset_after(self, mock_h5py_file):
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.keys.return_value = ['timestamps']
        mock_file.__getitem__.side_effect = {'timestamps': np.array(self.timestamps)}.__getitem__
        mock_h5py_file.return_value = mock_file
        
        with h5py.File('test_file.h5', 'r+') as h5dataset:
            add_annotations(h5dataset, [1609459211], self.preictal_duration, self.prediction_latency)

        expected_labeling = [False, False, False, False, False, True, True, True, True, True, False]

        args, kwargs = mock_file.create_dataset.call_args
        self.assertEqual(args[0], 'annotations') 
        np.testing.assert_array_equal(kwargs['data'], expected_labeling)


    @patch("h5py.File", autospec=True)
    def test_add_sz_onsets(self, mock_h5py_file):
        # Mock HDF5 file behavior
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_h5py_file.return_value = mock_file
        
        with h5py.File('test_file.h5', 'r+') as h5dataset:
            add_sz_onsets(h5dataset, self.sz_onsets_ts)

        # Check if the dataset was correctly
        args, kwargs = mock_file.create_dataset.call_args
        self.assertEqual(args[0], 'sz_onsets')  # Check dataset name
        np.testing.assert_array_equal(kwargs['data'], self.sz_onsets_ts)   # Check data content

if __name__ == '__main__':
    unittest.main()