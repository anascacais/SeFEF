# -*- coding: utf-8 -*-
"""
sefef.scoring
-------------

This module contains functions to compute both deterministic and probvabilistic metrics according to the horizon of the forecast.

:copyright: (c) 2024 by Ana Sofia Carmo
:license: BSD 3-clause License, see LICENSE for more details.
"""

# third-party
import pandas as pd
import numpy as np
import sklearn
import sklearn.metrics
import plotly.graph_objects as go


# local
from sefef.visualization import COLOR_PALETTE


class Scorer:
    ''' Class description 

    Attributes
    ----------  
    metrics2compute : list<str>
        List of metrics to compute. The metrics can be either deterministic or probabilistic and metric names should be the ones from the following list:
        - Deterministic: "Sen" (i.e. sensitivity), "FPR" (i.e. false positive rate), "TiW" (i.e. time in warning), "AUC_TiW" (i.e. area under the curve of Sen vs TiW). 
        - Probabilistic: "resolution", "reliability", "BS" (i.e. Brier score), "skill" or "BSS" (i.e. Brier skill score).    
    sz_onsets : array-like, shape (#seizures, ), dtype "int64"
        Contains the Unix timestamps, in seconds, for the start of each seizure onset.
    forecast_horizon : int
        Forecast horizon in seconds, i.e. time in the future for which the forecasts are valid.  
    performance : dict
        Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance. It is initialized as an empty dictionary and populated in "compute_metrics".
    reference_method : str, defaults to "prior_prob"
        Method to compute the reference forecasts.
    hist_prior_prob : float64, defaults to None
        Prior probability, aka historical likelihood (relative frequency) of seizures in train data. Used only as the "hist_prior_prob" reference forecast compute the skill measure. 
    Methods
    -------
    compute_metrics(forecasts, timestamps):
        Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute. This method uses techniques described in [Mason2004]_ and [Stephenson2008]_. 
    reliability_diagram() :
        Description

    Raises
    -------
    ValueError :
        Raised when a metric name in "metrics2compute" is not a valid metric or when "reference_method" is not a valid method.
    AttributeError :
        Raised when 'compute_metrics' is called before 'compute_metrics'.

    References
    ----------
    .. [Mason2004] S. J. Mason, “On Using ‘Climatology’ as a Reference Strategy in the Brier and Ranked Probability Skill Scores,” Jul. 2004, Accessed: Nov. 06, 2024. [Online]. Available: https://journals.ametsoc.org/view/journals/mwre/132/7/1520-0493_2004_132_1891_oucaar_2.0.co_2.xml
    .. [Stephenson2008] Stephenson, D. B. , C. A. S. Coelho, and I. T. Jolliffe. "Two Extra Components in the Brier Score Decomposition", Weather and Forecasting 23, 4 (2008): 752-757, doi: https://doi.org/10.1175/2007WAF2006116.1
    '''

    def __init__(self, metrics2compute, sz_onsets, forecast_horizon, reference_method='prior_prob', hist_prior_prob=None):
        self.metrics2compute = metrics2compute
        self.sz_onsets = np.array(sz_onsets)
        self.forecast_horizon = forecast_horizon
        self.reference_method = reference_method
        self.hist_prior_prob = hist_prior_prob
        self.performance = {}

        self.metrics2function = {'Sen': self._compute_Sen, 'FPR': self._compute_FPR, 'TiW': self._compute_TiW, 'AUC_TiW': self._compute_AUC, 'resolution': self._compute_resolution,
                                 'reliability': self._compute_reliability, 'calibration': self._compute_reliability, 'BS': self._compute_BS,  'skill': self._compute_skill, 'BSS': self._compute_skill}

    def compute_metrics(self, forecasts, timestamps, threshold=0.5, binning_method='quantile', num_bins=10, draw_diagram=True):
        ''' Computes metrics in "metrics2compute" for the probabilities in "forecasts" and populates the "performance" attribute.

        Parameters
        ---------- 
        forecasts : array-like, shape (#forecasts, ), dtype "float64"
            Contains the predicted probabilites of seizure occurrence for the period with duration equal to the forecast horizon and starting at the timestamps in "timestamps".
        timestamps : array-like, shape (#forecasts, ), dtype "int64"
            Contains the Unix timestamps, in seconds, for the start of the period for which the forecasts (in "forecasts") are valid. 
        threshold : float64, defaults to 0.5
            Probability value to apply as the high-likelihood threshold. 
        binning_method : str, defaults to "equal_frequency"
            Method used to determine the number of bins used to compute probabilistic metrics. Available methods are: 
                - "uniform": number of bins corresponds to np.ceil(#forecasts^(1/3)), set at approximately equal distances.
                - "quantile": number of bins corresponds to np.ceil(#forecasts^(1/3)), which are populated with an approximately equal number of forecasts.
        num_bins : int64, defaults to 10
            Number of bins used to compute probabilistic metrics. If None, it is calculated as np.ceil(#forecasts^(1/3)), otherwise "num_bins" is used as the number of bins.
        draw_diagram : bool, defaults to True
            Whether to draw the reliability diagram after computing all required metrics. 
        Returns
        -------
        performance : dict 
            Dictionary where the keys are the metrics' names (as in "metrics2compute") and the value is the corresponding performance.
        '''

        timestamps = np.array(timestamps)
        forecasts = np.array(forecasts)

        timestamps = timestamps[~np.isnan(forecasts)]
        forecasts = forecasts[~np.isnan(forecasts)]  # TODO: VERIFY THIS

        for metric_name in self.metrics2compute:
            if metric_name in ['Sen', 'FPR', 'TiW']:
                tp, fp, fn = self._get_counts(forecasts, timestamps, threshold)
                self.performance[metric_name] = self.metrics2function[metric_name](tp, fp, fn, forecasts)
            elif metric_name == 'AUC_TiW':
                self.performance[metric_name] = self.metrics2function[metric_name](forecasts, timestamps, threshold)
            elif metric_name in ['resolution', 'reliability', 'calibration', 'BS', 'skill', 'BSS']:
                bin_edges = self._get_bins_indx(forecasts, binning_method, num_bins)
                self.performance[metric_name] = self.metrics2function[metric_name](forecasts, timestamps, bin_edges)
            else:
                raise ValueError(f'{metric_name} is not a valid metric.')

        if draw_diagram:
            self.reliability_diagram(forecasts, timestamps, binning_method=binning_method, num_bins=num_bins)

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

        tp, fp, fn = np.vectorize(self._get_counts, excluded=['forecasts', 'timestamps_start_forecast'])(
            forecasts=forecasts, timestamps_start_forecast=timestamps, threshold=thresholds)

        sen = np.vectorize(self._compute_Sen, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)
        tiw = np.vectorize(self._compute_TiW, excluded=['forecasts'])(tp=tp, fp=fp, fn=fn, forecasts=forecasts)

        # add point (0, 0) to curve since the auc() function computes the area strictly based on the given points
        return sklearn.metrics.auc(np.append(tiw, 0.), np.append(sen, 0.))

    # Probabilistic metrics

    def _get_bins_indx(self, forecasts, binning_method, num_bins):
        '''Internal method that computes the edges of probability bins so that each bin contains the same number of observations. If not provided, the number of bins is determined by n^(1/3), as proposed in np.histogram_bin_edges.'''

        if num_bins is None:
            num_bins = np.ceil(len(forecasts)**(1/3)).astype('int64')

        if binning_method == 'uniform':
            bin_edges = np.linspace(0, 1, num_bins + 1)
        elif binning_method == 'quantile':
            percentile = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(np.sort(forecasts), percentile)[1:]  # remove edge corresponding to 0th percentile
        else:
            raise ValueError(f'{binning_method} is not a valid binning method')

        return bin_edges

    def _compute_resolution(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes the resolution, i.e. the ability of the model to diﬀerentiate between individual observed probabilities and the average observed probability. "y_avg": observed relative frequency of true events for all forecasts; "y_k_avg": observed relative frequency of true events for the kth probability bin.'''

        binned_data = np.digitize(forecasts, bin_edges, right=True)
        y_avg = len(self.sz_onsets) / len(forecasts)
        resolution = []

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data == k)
            events_in_bin, _, _ = self._get_counts(forecasts[binned_indx], timestamps[binned_indx], threshold=0.)
            y_k_avg = events_in_bin / len(forecasts[binned_indx])
            resolution += [len(forecasts[binned_indx]) * ((y_k_avg - y_avg) ** 2)]

        return np.sum(resolution) * (1/len(forecasts))

    def _compute_reliability(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes reliability, i.e. the agreement between forecasted and observed probabilities through the Brier score. "y_k_avg": observed relative frequency of true events for the kth probability bin.'''

        binned_data = np.digitize(forecasts, bin_edges, right=True)
        reliability = []

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data == k)
            events_in_bin, _, _ = self._get_counts(forecasts[binned_indx], timestamps[binned_indx], threshold=0.)
            y_k_avg = events_in_bin / len(forecasts[binned_indx])
            reliability += [len(forecasts[binned_indx]) * ((np.mean(forecasts[binned_indx]) - y_k_avg) ** 2)]

        return np.sum(reliability) * (1/len(forecasts))

    def _compute_uncertainty(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes uncertainty. "y_avg": observed relative frequency of true events for all forecasts'''
        y_avg = len(self.sz_onsets) / len(forecasts)
        return y_avg * (1-y_avg)

    def _compute_WBV(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes within-bin variance.'''
        binned_data = np.digitize(forecasts, bin_edges, right=True)
        wbv = []

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data == k)
            wbv += [np.sum((forecasts[binned_indx] - np.mean(forecasts[binned_indx]))**2)]

        return np.sum(wbv) * (1/len(forecasts))

    def _compute_WBC(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes within-bin covariance.'''
        binned_data = np.digitize(forecasts, bin_edges, right=True)
        wbc = []

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data == k)
            timestamps_start_forecast = timestamps[binned_indx]
            timestamps_end_forecast = timestamps_start_forecast + self.forecast_horizon - 1
            y_ki = np.any((self.sz_onsets[:, np.newaxis] >= timestamps_start_forecast[np.newaxis, :]) & (
                self.sz_onsets[:, np.newaxis] <= timestamps_end_forecast[np.newaxis, :]), axis=0)
            wbc += [np.sum((y_ki - np.mean(y_ki)) * (forecasts[binned_indx] - np.mean(forecasts[binned_indx])))]

        return np.sum(wbc) * (2/len(forecasts))

    def _compute_BS(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes the Brier score, through the decomposition proposed in [Stephenson2008]_.'''

        if 'reliability' in self.performance.keys():
            reliability = self.performance['reliability']
        else:
            reliability = self._compute_reliability(forecasts, timestamps, bin_edges)

        if 'resolution' in self.performance.keys():
            resolution = self.performance['resolution']
        else:
            resolution = self._compute_resolution(forecasts, timestamps, bin_edges)

        uncertainty = self._compute_uncertainty(forecasts, timestamps, bin_edges)
        wbv = self._compute_WBV(forecasts, timestamps, bin_edges)
        wbc = self._compute_WBC(forecasts, timestamps, bin_edges)

        return (reliability - resolution + uncertainty + wbv - wbc)

    def _compute_skill(self, forecasts, timestamps, bin_edges):
        '''Internal method that computes the Brier skill score against a reference forecast. Simplification of BS of reference forecast as described in [Mason2004]_.'''

        if 'BS' in self.performance.keys():
            bs = self.performance['BS']
        else:
            bs = self._compute_BS(forecasts, timestamps, bin_edges)

        ref_forecasts = self._get_reference_forecasts(timestamps)
        return 1 - bs / self._compute_uncertainty(ref_forecasts, None, None)

    def _get_reference_forecasts(self, timestamps):
        '''Internal method that returns a reference forecast according to the specified method. "y_avg": observed relative frequency of true events for all forecasts.'''
        if self.reference_method == 'prior_prob':
            y_avg = len(self.sz_onsets) / len(timestamps)
            return y_avg * np.ones_like(timestamps)
        else:
            raise ValueError(f'{self.reference_method} is not a valid method to compute the reference forecasts.')

    def reliability_diagram(self, forecasts, timestamps, binning_method, num_bins):
        '''Method that plots the reliability diagram (forecasted_proba vs observed_proba), along with the no-resolution and perfect-reliability lines.'''
        fig = go.Figure()

        y_avg = len(self.sz_onsets) / len(forecasts)

        bin_edges = self._get_bins_indx(forecasts, binning_method, num_bins)
        binned_data = np.digitize(forecasts, bin_edges, right=True)
        bin_edges = np.insert(bin_edges, 0, 0.)

        diagram_data = pd.DataFrame(columns=['observed_proba', 'forecasted_proba'],
                                    index=(bin_edges[:-1] + bin_edges[1:]) / 2)

        for k in np.unique(binned_data):
            binned_indx = np.where(binned_data == k)
            events_in_bin, _, _ = self._get_counts(
                forecasts[binned_indx], timestamps[binned_indx], threshold=0.)
            y_k_avg = events_in_bin / len(forecasts[binned_indx])
            diagram_data.iloc[k, :] = [y_k_avg, np.mean(forecasts[binned_indx])]

        fig.add_trace(go.Scatter(
            x=diagram_data.loc[:, 'forecasted_proba'],
            y=diagram_data.loc[:, 'observed_proba'],
            mode='lines',
            line=dict(width=3, color=COLOR_PALETTE[1]),
            name='Reliability curve'
        ))

        fig.add_trace(go.Scatter(
            x=diagram_data.loc[:, 'forecasted_proba'],
            y=diagram_data.loc[:, 'observed_proba'],
            mode='markers',
            marker=dict(size=10, color=COLOR_PALETTE[1]),
            name='Bin average'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(width=3, color=COLOR_PALETTE[0], dash='dash'),
            # showlegend=False,
            mode='lines',
            name='Perfect reliability'
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[y_avg, y_avg],
            line=dict(width=3, color='lightgrey', dash='dash'),
            mode='lines',
            name='No resolution'
        ))

        # Config plot layout
        fig.update_yaxes(
            title='observed probability',
            tickfont=dict(size=12),
            showline=True, linewidth=2, linecolor=COLOR_PALETTE[2],
            showgrid=False,
            range=[diagram_data.min().min(), diagram_data.max().max()]
        )
        fig.update_xaxes(
            title='forecasted probability',
            tickfont=dict(size=12),
            showline=True, linewidth=2, linecolor=COLOR_PALETTE[2],
            showgrid=False,
            range=[diagram_data.min().min(), diagram_data.max().max()],
        )
        fig.update_layout(
            title=f'Reliability diagram (binning method: {binning_method})',
            showlegend=True,
            plot_bgcolor='white',
        )
        fig.show()
