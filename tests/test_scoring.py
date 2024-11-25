import unittest
import numpy as np

from sefef.scoring import Scorer

class TestScorer(unittest.TestCase):

    def setUp(self):
        self.sz_onsets 
        self.forecast_horizon 
        self.reference_method = 'prior_prob'
        self.hist_prior_prob 
        self.forecasts
        self.timestamps
        self.binning_method
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
        metrics2compute = ['sensitivity']
        scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
        with self.assertRaises(ValueError):
            _ = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)
    
    # def test_sensitivity(self):
    #     metrics2compute = ['Sen']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


    # def test_false_positive_rate(self):
    #     metrics2compute = ['FPR']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


    # def test_time_in_warning(self):
    #     metrics2compute = ['TiW']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)

    
    # def test_AUC(self):
    #     metrics2compute = ['AUC']
    #     scorer = Scorer(metrics2compute, self.sz_onsets, self.forecast_horizon, self.reference_method, self.hist_prior_prob)
    #     performance = scorer.compute_metrics(self.forecasts, self.timestamps, binning_method=self.binning_method, num_bins=self.num_bins, draw_diagram=False)


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