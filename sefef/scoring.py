

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
        Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value. It is initialized as an empty dictionary and populated in "compute_metrics0".
    
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

    def __init__(self, metrics2compute, sz_onsets):
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
        result : bool
            Description
        ''' 
        pass