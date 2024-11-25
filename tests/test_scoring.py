import unittest
import numpy as np

from sefef.scoring import Scorer

class TestScorer(unittest.TestCase):

    def setUp(self):
        self.sz_onsets = [1609459860]
        self.forecast_horizon = 6*60
        self.reference_method = 'prior_prob'
        self.hist_prior_prob = 1/4
        self.forecasts_sample_time = np.array([1, 1, 0], dtype='float64')
        self.forecasts_clock_time = np.array([0.5, 1, 0], dtype='float64')
        self.timestamps = np.arange(1609459620, 1609460460, self.forecast_horizon)
        self.binning_method = 'equal_frequency'
        self.num_bins = 10

        self.scorer = Scorer([], self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)

    
    # 1. Test Initialization:  Verify that attributes are correctly initialized.
    def test_initialization(self):
        self.assertTrue(isinstance(self.scorer.metrics2compute, list))
        self.assertTrue(isinstance(self.scorer.sz_onsets, np.ndarray))
        self.assertTrue(isinstance(self.scorer.forecast_horizon, int))
        self.assertTrue(isinstance(self.scorer.reference_method, str))
        self.assertTrue(isinstance(self.scorer.hist_prior_prob, float))
        self.assertTrue(isinstance(self.scorer.performance, dict))

    
    def test_non_existent_metric(self):
        metrics2compute = ['Sens']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        with self.assertRaises(ValueError):
            _ = scorer.compute_metrics(self.forecasts_sample_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
    

    def test_Sen_label_as_prediction_w_sample_time(self):
        metrics2compute = ['Sen']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_sample_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 1 
        self.assertTrue(performance['Sen'] == expected_performance)


    def test_Sen_label_as_prediction_w_clock_time(self):
        metrics2compute = ['Sen']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_clock_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 1
        self.assertTrue(performance['Sen'] == expected_performance)


    def test_FPR_label_as_prediction_w_sample_time(self):
        metrics2compute = ['FPR']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_sample_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 1/3
        self.assertTrue(performance['FPR'] == expected_performance)


    def test_FPR_label_as_prediction_w_clock_time(self):
        metrics2compute = ['FPR']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_clock_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 1/3
        self.assertTrue(performance['FPR'] == expected_performance)


    def test_TiW_label_as_prediction_w_sample_time(self):
        metrics2compute = ['TiW']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_sample_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 2/3
        self.assertTrue(performance['TiW'] == expected_performance)


    def test_TiW_label_as_prediction_w_clock_time(self):
        metrics2compute = ['TiW']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_clock_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 2/3
        self.assertTrue(performance['TiW'] == expected_performance)

    
    def test_AUC_tiw_label_as_prediction_w_sample_time(self):
        metrics2compute = ['AUC-TiW']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_sample_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 2/3
        self.assertTrue(performance['AUC-TiW'] == expected_performance)


    def test_AUC_tiw_label_as_prediction_w_clock_time(self):
        metrics2compute = ['AUC-TiW']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        performance = scorer.compute_metrics(self.forecasts_clock_time, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
        expected_performance = 0.5
        self.assertTrue(performance['AUC-TiW'] == expected_performance)


    # def test_resolution(self):
    #     metrics2compute = ['resolution']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


    # def test_reliability(self):
    #     metrics2compute = ['reliability']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


    # def test_brier_score(self):
    #     metrics2compute = ['BS']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


    # def test_brier_skill_score(self):
    #     metrics2compute = ['BSS']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)



if __name__ == '__main__':
    unittest.main()