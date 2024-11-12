# third-party
import pandas as pd


class Forecast:
    ''' Stores the forecasts made by the model and processes them.

    Attributes
    ---------- 
    pred_proba : array-like, shape (#samples, ), dtype "float64"
        Contains the probability predicted by the model for each sample belonging to the pre-ictal class. 
    timestamps :  array-like, shape (#samples, ), dtype "Int64"
        Contains the unix timestamps (in seconds) corresponding to the start-time of each sample. 

    Methods
    -------
    append(pred_proba, timestamps) :
        Appends new predicted probabilities to the ones already in the Forecast object.
    postprocess(forecast_horizon) :
        Applies postprocessing methodology to the predictions stored in "pred_proba", according to "forecast horizon" (in seconds). Returns an array with the new probabilities.

    Raises
    -------
    ValueError :
        Description
    '''

    def __init__(self, pred_proba, timestamps):
        self.pred_proba = pred_proba
        self.timestamps = timestamps

    def append(self, pred_proba, timestamps):
        pass

    def postprocess(self, forecast_horizon, smooth_win, origin='clock-time'):
        ''' Applies post-processing methodology to the predictions stored in "pred_proba". For each time period with duration equal to "forecast_horizon", mean predicted probabilities are calculated for groups of consecutive samples (with a window of duration "smooth_win", in seconds), without overlap, and the maximum across the full period is obtained.

        Parameters
        ---------- 
        forecast_horizon : int
            Forecast horizon in seconds, i.e. time in the future for which the forecasts will be issued.  
        smooth_win : int
            Duration of window, in seconds, used to smooth the predicted probabilities (without overlap). 
        origin : str, defaults to "clock-time"
            Determines if the forecasts are issued at clock-time (e.g. at the start of each hour) or according to the start-time of the first sample. Options are "clock-time" and "sample-time", respectively. 

        Returns
        -------
        result1 : array-like, shape (#forecasts, ), dtype "float64"
            Contains the predicted probabilites of seizure occurrence for the period with duration "forecast_horizon" and starting at the timestamps in "result2".
        result2 : array-like, shape (#forecasts, ), dtype "Int64"
            Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "result1") are valid. 
        '''
        origin2param = {'clock-time': 'start_day', 'sample-time': 'start'}

        pred_proba = pd.DataFrame(self.pred_proba, index=pd.to_datetime(
            self.timestamps, unit='s', utc=True), columns=['pred_proba'])
        smooth_proba = pred_proba.resample(f'{smooth_win}s', origin=origin2param[origin]).mean()

        final_proba = smooth_proba.resample(f'{forecast_horizon}s', origin=origin2param[origin]).max()
        final_proba.index = final_proba.index + pd.Timedelta(f'{forecast_horizon}s')

        return final_proba.pred_proba.to_numpy(), (final_proba.index.astype('Int64') // 10**9).to_numpy()
