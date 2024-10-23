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
    reference_method : str, defaults to "naive"
        Method to compute the reference forecasts.
    Methods
    -------
    compute_metrics(forecasts, timestamps):
        Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.
    reliability_diagram() :
        Description
    
    Raises
    -------
    ValueError :
        Raised when a metric name in "metrics2compute" is not a valid metric or when "reference_method" is not a valid method.
    AttributeError :
        Raised when 'compute_metrics' is called before 'compute_metrics'.
    ''' 

    def __init__(self, metrics2compute, sz_onsets, forecast_horizon, reference_method='naive'):
        self.metrics2compute = metrics2compute
        self.sz_onsets = sz_onsets
        self.forecast_horizon = forecast_horizon
        self.reference_method = reference_method
        self.performance = {}

    def compute_metrics(self, forecasts, timestamps, threshold=0.5, binning_method='auto'):
        ''' Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.
        
        Parameters
        ---------- 
        forecasts : array-like, shape (#forecasts, ), dtype "float64"
            Contains the predicted probabilites of seizure occurrence for the period with duration equal to the forecast horizon and starting at the timestamps in "timestamps".
        timestamps : array-like, shape (#forecasts, ), dtype "int64"
            Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "forecasts") are valid. 
        threshold : float64, defaults to 0.5
            Probability value to apply as the high-likelihood threshold. 
        binning_method : str, defaults to "auto"
            Method used to determine the number of bins. Currently only supports the default method, which corresponds to np.ceil(#forecasts^(1/3)), which are populated with an approximately equal number of observations.
        Returns
        -------
        performance : dict 
            Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance.
        ''' 

        timestamps = timestamps[~np.isnan(forecasts)]
        forecasts = forecasts[~np.isnan(forecasts)] # TODO: VERIFY THIS 

        metrics2function = {'Sen': self._compute_Sen, 'FPR': self._compute_FPR, 'TiW': self._compute_TiW, 'AUC': self._compute_AUC, 'resolution': self._compute_resolution, 'reliability': self._compute_reliability, 'BS': self._compute_reliability, 'skill': self._compute_skill, 'BSS': self._compute_skill}
                    
        for metric_name in self.metrics2compute:
            if metric_name in ['Sen', 'FPR', 'TiW']:
                tp, fp, fn = self._get_counts(forecasts, timestamps, threshold)
                self.performance[metric_name] = metrics2function[metric_name](tp, fp, fn, forecasts)
            elif metric_name == 'AUC':
                self.performance[metric_name] = metrics2function[metric_name](forecasts, timestamps, threshold)
            elif metric_name in ['resolution', 'reliability', 'BS', 'skill', 'BSS']:
                binned_data = self._get_bins_indx(forecasts, binning_method)
                self.performance[metric_name] = metrics2function[metric_name](forecasts, timestamps, binned_data)
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

        tp, fp, fn = np.vectorize(self._get_counts, excluded=['forecasts', 'timestamps_start_forecast'])(forecasts=forecasts, timestamps_start_forecast=timestamps, threshold=thresholds) 
        
        sen = np.vectorize(self._compute_Sen, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)
        tiw = np.vectorize(self._compute_TiW, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)
        
        return sklearn.metrics.auc(tiw, sen)

    

    # Probabilistic metrics
    def _get_bins_indx(self, forecasts, binning_method):
        '''Internal method that computes the edges of probability bins so that each bin contains the same number of observations. The number of bins is determined by n^(1/3), as proposed in np.histogram_bin_edges.'''
        if binning_method == 'auto':
            num_bins = np.ceil(len(forecasts)**(1/3)).astype('int64')
        else:
            raise NotImplementedError
        percentile = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(np.sort(forecasts), percentile)[1:]  #remove edge corresponding to 0th percentile
        binned_data = np.digitize(forecasts, bin_edges, right=True)
        return binned_data

    def _compute_resolution(self, forecasts, timestamps, binned_data):
        '''Internal method that computes the resolution, i.e. the ability of the model to diﬀerentiate between individual observed probabilities and the average observed probability. "y_avg": observed relative frequency of true events for all forecasts; "y_k_avg": observed relative frequency of true events for the kth probability bin.'''
        
        y_avg = len(self.sz_onsets) / len(forecasts)
        resolution = np.empty((len(np.unique(binned_data)),))

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data==k)
            events_in_bin, _, _ = self._get_counts(forecasts[binned_indx], timestamps[binned_indx], threshold=0.)
            y_k_avg = events_in_bin / len(binned_indx)
            resolution[k] = len(binned_indx) * ((y_k_avg - y_avg) ** 2)
        
        return np.sum(resolution) * (1/len(forecasts))

    def _compute_reliability(self, forecasts, timestamps, binned_data):
        '''Internal method that computes reliability, i.e. the agreement between forecasted and observed probabilities through the Brier score. "y_k_avg": observed relative frequency of true events for the kth probability bin.'''
        
        reliability = np.empty((len(np.unique(binned_data)),))

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data==k)
            events_in_bin, _, _ = self._get_counts(forecasts[binned_indx], timestamps[binned_indx], threshold=0.)
            y_k_avg = events_in_bin / len(binned_indx)
            reliability[k] = len(binned_indx) * ((np.mean(forecasts[binned_indx]) - y_k_avg) ** 2)

        return np.sum(reliability) * (1/len(forecasts))

    def _compute_skill(self, forecasts, timestamps, binned_data):
        '''Internal method that computes the Brier skill score against a reference forecast.'''
        skill = self._compute_reliability(forecasts, timestamps, binned_data)
        ref_forecasts = self._get_reference_forecasts(timestamps)
        ref_skill = self._compute_reliability(ref_forecasts, timestamps, binned_data)
        return 1 - skill / ref_skill


    def _get_reference_forecasts(self, timestamps):
        '''Internal method that returns a reference forecast according to the specified method. "y_avg": observed relative frequency of true events for all forecasts.'''
        if self.reference_method == 'naive':
            y_avg = len(self.sz_onsets) / len(timestamps)
            return y_avg * np.ones_like(timestamps)
        else:
            raise ValueError(f'{self.reference_method} is not a valid method to compute the reference forecasts.')