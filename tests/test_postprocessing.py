import unittest
import numpy as np

from sefef.postprocessing import Forecast

class TestForecast(unittest.TestCase):

    def setUp(self):
        self.pred_proba = [0]*4 + [1]*5 + [0]*11
        self.timestamps = np.arange(1609459260, 1609460460, 60).tolist()
        self.forecast = Forecast(self.pred_proba, self.timestamps)

        self.forecast_horizon = 6*60
        self.smooth_win = 2*60
    
    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.forecast.pred_proba, np.ndarray))
        self.assertTrue(isinstance(self.forecast.timestamps, np.ndarray))

    def test_incompatible_fh_sm(self):
        smooth_win = 2.5*60
        with self.assertRaises(AssertionError):
            _, _ = self.forecast.postprocess(self.forecast_horizon, smooth_win)

    def test_postprocess_sample_time(self):
        postprocessed, final_timestamps = self.forecast.postprocess(self.forecast_horizon, self.smooth_win, origin='sample-time')

        expected_postprocessed = np.array([1, 1, 0, 0], dtype='float64')
        expected_final_timestamps = (np.arange(1609459620, 1609461060, self.forecast_horizon)).tolist()

        np.testing.assert_array_equal(postprocessed, expected_postprocessed)
        np.testing.assert_array_equal(final_timestamps, expected_final_timestamps)

    def test_postprocess_clock_time(self):
        postprocessed, final_timestamps = self.forecast.postprocess(self.forecast_horizon, self.smooth_win, origin='clock-time')

        expected_postprocessed = np.array([0.5, 1, 0, 0], dtype='float64')
        expected_final_timestamps = (np.arange(1609459560, 1609460700, self.forecast_horizon)).tolist()

        np.testing.assert_array_equal(postprocessed, expected_postprocessed)
        np.testing.assert_array_equal(final_timestamps, expected_final_timestamps)

if __name__ == '__main__':
    unittest.main()