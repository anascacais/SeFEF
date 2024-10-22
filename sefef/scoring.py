

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

    def __init__(self, metrics2compute, sz_onsets, ):
        self.metrics2compute = metrics2compute
        self.sz_onsets = sz_onsets
        self.performance = {}

    def compute_metrics(self, forecasts, timestamps):
        ''' Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.
        
        Parameters
        ---------- 
        forecasts : array-like, shape (#forecasts, ), dtype "float64"
            Contains the predicted probabilites of seizure occurrence for the period with duration equal to the forecast horizon and starting at the timestamps in "timestamps".
        timestamps : array-like, shape (#forecasts, ), dtype "int64"
            Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "forecasts") are valid. 
        
        Returns
        -------
        performance : dict 
            Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance.
        ''' 

        metrics2function = {'Sen': self._compute_Sen, 'FPR': self._compute_FPR, 'TiW': self._compute_TiW, 'AUC': self._compute_AUC, 'resolution': self._compute_resolution, 'reliability': self._compute_BS, 'BS': self._compute_BS, 'skill': self._compute_BSS, 'BSS': self._compute_BSS}
                    
        for metric_name in self.metrics2compute:
            if metric_name in ['Sen', 'FPR', 'TiW', 'AUC']:
                tp, fp, fn = self._get_counts(forecasts, timestamps)
                self.performance[metric_name] = metrics2function[metric_name](tp, fp, fn)
            elif metric_name in ['resolution', 'reliability', 'BS', 'skill', 'BSS']:
                proba_bins = self._get_probs_bins(forecasts)
                self.performance[metric_name] = metrics2function[metric_name](proba_bins)
            else: 
                raise ValueError(f'{metric_name} is not a valid metric.')
        
        return self.performance

    # Deterministic metrics
    def _get_counts(self, forecasts, timestamps):
        ''''''
        pass

    def _compute_Sen(self, tp, fp, fn):
        ''' Internal method that ... ''' 
        pass

    def _compute_FPR(self, tp, fp, fn):
        ''' Internal method that ... ''' 
        pass

    def _compute_TiW(self, tp, fp, fn):
        ''' Internal method that ... ''' 
        pass

    def _compute_AUC(self, tp, fp, fn):
        ''' Internal method that ... ''' 
        pass

    

    # Probabilistic metrics
    def _get_probs_bins(self, forecasts):
        ''''''
        pass

    def _compute_resolution(self, proba_bins):
        ''' Internal method that ... ''' 
        pass

    def _compute_BS(self, proba_bins):
        ''' Internal method that ... ''' 
        pass

    def _compute_BSS(self, proba_bins):
        ''' Internal method that ... ''' 
        pass