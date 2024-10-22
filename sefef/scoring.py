# third-party
import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics

class Scorer:
    ''' Class description 
    
    Attributes
    ----------  
    metrics2compute : list<str>
        List of metrics to compute. The metrics can be either deterministic or probabilistic and metric names should be the ones from the following list:
        - Deterministic: "Sen" (i.e. sensitivity), "FPR" (i.e. false positive rate), "TiW" (i.e. time in warning), "AUC" (i.e. area under the ROC curve). 
        - Probabilistic: "resolution", "reliability" or "BS" (i.e. Brier score), "skill" or "BSS" (i.e. Brier skill score).    
    sz_onsets : array-like, shape (#seizures, ), dtype "int64"
        Contains the Unix timestamps, in seconds, for the start of each seizure onset.
    forecast_horizon : int
        Forecast horizon in seconds, i.e. time in the future for which the forecasts are valid.  
    performance : dict
        Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance. It is initialized as an empty dictionary and populated in "compute_metrics".
    
    Methods
    -------
    compute_metrics(forecasts, timestamps):
        Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.
    reliability_diagram() :
        Description
    
    Raises
    -------
    ValueError :
        Raised when a metric name in "metrics2compute" is not a valid metric. 
    AttributeError :
        Raised when 'compute_metrics' is called before 'compute_metrics'.
    ''' 

    def __init__(self, metrics2compute, sz_onsets, forecast_horizon):
        self.metrics2compute = metrics2compute
        self.sz_onsets = sz_onsets
        self.forecast_horizon = forecast_horizon
        self.performance = {}

    def compute_metrics(self, forecasts, timestamps, threshold=0.5):
        ''' Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.
        
        Parameters
        ---------- 
        forecasts : array-like, shape (#forecasts, ), dtype "float64"
            Contains the predicted probabilites of seizure occurrence for the period with duration equal to the forecast horizon and starting at the timestamps in "timestamps".
        timestamps : array-like, shape (#forecasts, ), dtype "int64"
            Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "forecasts") are valid. 
        threshold : float64, defaults to 0.5
            Probability value to apply as the high-likelihood threshold. 
            
        Returns
        -------
        performance : dict 
            Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance.
        ''' 

        metrics2function = {'Sen': self._compute_Sen, 'FPR': self._compute_FPR, 'TiW': self._compute_TiW, 'AUC': self._compute_AUC, 'resolution': self._compute_resolution, 'reliability': self._compute_BS, 'BS': self._compute_BS, 'skill': self._compute_BSS, 'BSS': self._compute_BSS}
                    
        for metric_name in self.metrics2compute:
            if metric_name in ['Sen', 'FPR', 'TiW']:
                tp, fp, fn = self._get_counts(forecasts, timestamps, threshold)
                self.performance[metric_name] = metrics2function[metric_name](tp, fp, fn, forecasts)
            elif metric_name == 'AUC':
                self.performance[metric_name] = metrics2function[metric_name](forecasts, timestamps, threshold)
            elif metric_name in ['resolution', 'reliability', 'BS', 'skill', 'BSS']:
                proba_bins = self._get_probs_bins(forecasts)
                self.performance[metric_name] = metrics2function[metric_name](proba_bins)
            else: 
                raise ValueError(f'{metric_name} is not a valid metric.')
        
        return self.performance

    # Deterministic metrics
    def _get_counts(self, forecasts, timestamps_start_forecast, threshold):
        '''Internal method that computes counts of true positives (tp), false positives (fp), and false negatives (fn), according to the occurrence (or not) of a seizure event within the forecast horizon.'''
        timestamps_end_forecast = timestamps_start_forecast + self.forecast_horizon - 1 
        
        tp_counts = np.any(
            (self.sz_onsets[:, np.newaxis] >= timestamps_start_forecast[np.newaxis, :]) 
            & (self.sz_onsets[:, np.newaxis] <= timestamps_end_forecast[np.newaxis, :])
            & (forecasts >= threshold),
            axis=1)

        no_sz_forecasts = forecasts[~np.any(
            (self.sz_onsets[:, np.newaxis] >= timestamps_start_forecast[np.newaxis, :]) 
            & (self.sz_onsets[:, np.newaxis] <= timestamps_end_forecast[np.newaxis, :]), 
            axis=0)]
        
        tp = np.sum(tp_counts)
        fn = len(self.sz_onsets) - tp
        fp = np.sum(no_sz_forecasts >= threshold)

        return tp, fp, fn

    def _compute_Sen(self, tp, fp, fn, forecasts):
        '''Internal method that computes sensitivity, providing a measure of the model's ability to correctly identify pre-ictal periods.'''
        return tp / (tp + fn)

    def _compute_FPR(self, tp, fp, fn, forecasts):
        '''Internal method that computes the false positive rate, i.e. the proportion of time that the user incorrectly spends in alert.'''
        return fp / len(forecasts)

    def _compute_TiW(self, tp, fp, fn, forecasts):
        '''Internal method that computes the time in warning, i.e. the proportion of time that the user spends in alert (i.e. in a high likelihood state, independently of the ”goodness” of the forecast).'''
        return (tp + fp) / len(forecasts)

    def _compute_AUC(self, forecasts, timestamps, threshold):
        '''Internal method that computes the area under the Sen vs TiW curve, abstracting the need for threshold optimization. Computed as the numerical integration of Sen vs TiW using the trapezoidal rule.'''
        # use unique forecasted values as thresholds
        thresholds = np.unique(forecasts)
        thresholds = thresholds[~np.isnan(thresholds)]

        tp, fp, fn = np.vectorize(self._get_counts, excluded=['forecasts', 'timestamps_start_forecast'])(forecasts=forecasts, timestamps_start_forecast=timestamps, threshold=thresholds) 
        
        sen = np.vectorize(self._compute_Sen, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)
        tiw = np.vectorize(self._compute_TiW, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)
        
        return sklearn.metrics.auc(tiw, sen)

    

    # Probabilistic metrics
    def _get_probs_bins(self, forecasts):
        ''''''
        pass

    def _compute_resolution(self, proba_bins):
        '''Internal method that ...'''
        pass

    def _compute_BS(self, proba_bins):
        '''Internal method that ...'''
        pass

    def _compute_BSS(self, proba_bins):
        '''Internal method that ...'''
        pass