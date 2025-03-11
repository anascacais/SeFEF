# third-party
import numpy as np
import pandas as pd
from scipy.stats import vonmises
import scipy
from scipy.optimize import minimize

# local
from seizureforecast.visualization import plot_event_phase_dist


class EventProbabilityEstimator:
    ''' Estimator of event probability given a circular statistics method. Significant cycles are identified and probabilities associated to each cycle are joined through the geometric mean of odds.

    Attributes
    ---------- 
    significant_cycles : list of arrays, shape of arrays (#cycles,), dtype=int64
        List of periods (in seconds) of significant cycles. Each element of the list corresponds to a window of analysis.
    '''

    def __init__(self):
        self.significant_cycles = None
        self.si_thr = None

    def _compute_bin_counts(self, phases, bin_edges):
        counts, _ = np.histogram(phases, bins=bin_edges)
        return counts

    def _compute_bin_phase_freq(self, event_phases, sample_phases, alpha, bin_edges):
        '''Internal method that, given the events' and all samples' phases for a certain cycle period, returns the frequency of events for the bins.'''
        event_counts, _ = np.histogram(event_phases, bins=bin_edges)
        bin_counts, _ = np.histogram(sample_phases, bins=bin_edges)
        smooth_bin_counts = bin_counts + alpha * len(bin_edges)
        smooth_bin_counts[smooth_bin_counts == 0] = 1
        freq = (event_counts + alpha) / smooth_bin_counts
        return freq

    def _compute_instantaneuos_phases(self, ts, cycles):
        '''Internal method that computes instantaneous phases of samples (#samples, ) for given cycle periods. Output (#samples, #cycles) is in radians. '''
        ts, cycles = np.array(ts), np.array(cycles)
        return (2 * np.pi * (ts[:, np.newaxis] % cycles[np.newaxis, :]) / cycles[np.newaxis, :])

    def _get_SI_significant_cycles(self, labels, ts, candidate_cycles):
        ''' Internal method that, from a set of candidate cycles, identifies the ones that are significant, i.e. that present a phase-locking with event occurrences with SI >= "si_thr". The list is ordered according to SI value.'''
        # Test only candidate cycles that are smaller than 1/2 of total duration of data
        candidate_cycles = self._validate_candidate_cycles(
            ts, candidate_cycles)

        # Compute instantaneous phase with respect to all candidate cycles
        phases = self._compute_instantaneuos_phases(
            ts[labels == True], candidate_cycles)
        # Get magnitude of mean vector (aka SI, aka PLV)
        si = np.abs(np.mean(np.exp(1j * phases), axis=0))

        # Indentify significant cycles
        significant_cycles = candidate_cycles[np.where(si >= self.si_thr)]
        significant_cycles = [val for _, val
                              in sorted(zip(si, significant_cycles), reverse=True)]
        significant_cycles = exclude_harmonics(significant_cycles)

        if len(significant_cycles) == 0:
            raise ValueError('No significant cycles were found.')

        return significant_cycles

    def _validate_candidate_cycles(self, ts, candidate_cycles):
        '''Internal method that returns only the candidate cycles that are smaller than 1/2 of total duration of data.'''
        total_duration = ts[-1] - ts[0]
        return candidate_cycles[candidate_cycles < 0.5 * total_duration]

    def _compute_combined_probability(self, cycle_based_prob):
        '''Internal method that computed the combined probability resulting from multiple cycles, through odds aggregation. Input has shape (#cycles, #samples).'''
        # Compute geometric mean of odds, following Berkson (1944) the sum of the log-odds ratios, i.e. the logits, of P:
        geometric_mean_odds = np.prod((
            cycle_based_prob / (1-cycle_based_prob)) ** (1/cycle_based_prob.shape[0]), axis=0)
        # Transform back to probability
        prob = np.divide(geometric_mean_odds, 1+geometric_mean_odds)
        return prob

    def _weighted_average_combined_probability(self, windowed_prob, time_diff, decay_factor=1.46E-7):
        '''Combine windowed probabilities using a weighted average approach. The weights decrease as the time difference between the timestamps and the test timestamps increase. An exponential decay function is used. Decay parameter is set so that the weight becomes negligible (0.01) after approximately 1 year. Both inputs have shape (#windows, #samples).'''
        weights = np.exp(-decay_factor * np.array(time_diff))
        normalized_weights = weights / np.sum(weights, axis=0)
        return np.sum(normalized_weights * windowed_prob, axis=0)

    def _get_windowed_data(self, ts, window_duration=None, window_overlap=None):
        '''Internal method that splits input timestamps into windows of duration "window_duration" (in seconds) equivalent to a moving window approach with overlap of duration "window_overlap" (in seconds). In case of missing periods of data larger than 1/2 "window_duration", data is split and analysed separately. '''

        if window_overlap is None:
            window_overlap = int(window_duration * 0.5)

        splits = np.split(ts, np.where(np.diff(ts) > window_duration/2)[0]+1)

        ts_windows_start, ts_windows_end = [], []

        for ts_split in splits:
            ts_windows_start += [ts_split[0]]
            while True:
                try:
                    ts_windows_end += [
                        ts_split[np.where(ts_split-ts_windows_start[-1] >= window_duration)[0][0]]]
                    ts_windows_start += [
                        ts_split[np.where(ts_windows_end[-1] - ts_split - window_overlap >= 0)[0][-1]]]
                except IndexError:
                    break

            if len(ts_windows_start) > len(ts_windows_end):
                ts_windows_start = ts_windows_start[:-1]

        return ts_windows_start, ts_windows_end


class VonMisesEstimator(EventProbabilityEstimator):
    ''' Estimator of event probability based on a von Mises distribution that is fitted to the train data.

    Attributes
    ---------- 
    forecast_horizon : int
        Forecast horizon in seconds.  
    significant_cycles : list of arrays, shape of arrays (#cycles,), dtype=int64
        List of periods (in seconds) of significant cycles. Each element of the list corresponds to a window of analysis.
    candidate_cycles : array-like, shape (#cycles,), dtype=int64
        Contains periods (in seconds) of candidate cycles.
    si_thr : float
        Minimum value of Synchronization Index (SI) to consider a cycle as significant (phase-locking with event occurrences).
    von_mises_params : list of arrays, shape of arrays (#cycles, 2)
        Contains the von Mises distribution paramaters (kappa, mu) for each signigicant cycle (according to the train window). Each element of the list corresponds to a window of analysis.
    marginal_phase_dist : list of arrays, shape of arrays (#cycles, num_bins)
        Contains the marginal phase distribution for the binned phases for each signigicant cycle (according to the train window). Each element of the list corresponds to a window of analysis.
    seizure_priors : list of floats
        Contains the seizure prior. Each element of the list corresponds to a window of analysis.
    ts_windows_end : array-like (#windows,)
        Last timestamp of each train window. This is used to assert how relevant each particular cycle may be to the test timestamp (according to how long ago it was). 
    window_duration : int, defaults to the total duration of the train set
        Duration (in seconds) of window to analyse.

    Methods
    -------
    train(train_ts, train_labels) :
        For each significant cycle, MLE is used to fit a von Mises distribution to the event data in train. 
    predict(test_ts)
        For each significant cycle, use the estimated distribution to compute sample probability. Probabilities associated to each significant cycle are joined through the geometric mean of odds.

    Raises
    -------
    ValueError :
        Raised when no sigificant cycles are found. 
    '''

    def __init__(self, forecast_horizon, laplace_smoothing=1):
        self.laplace_smoothing = laplace_smoothing
        self.window_duration = None
        self.forecast_horizon = forecast_horizon

        self.von_mises_dist_sz = []
        self.von_mises_dist_nonsz = []

    def train(self, train_ts, train_labels, candidate_cycles, si_thr, max_n_cycles=None, window_duration=None, **args):
        ''' Computes likelihoods for phase bins, according to significant cycles. Von Mises fitting using MLE with a self-defined negative log-likelihood (NLL).

        Parameters
        ---------- 
        train_ts : array-like, shape (#samples, ), dtype=int64
            Contains the unix timestamp (in seconds) of the samples.
        train_labels : array-like, shape (#samples, ), dtype=bool
            Contains the labels of each sample.
        candidate_cycles : array-like, shape (#cycles,), dtype=int64
            Contains periods (in seconds) of candidate cycles.
        si_thr : float
            Minimum value of Synchronization Index (SI) to consider a cycle as significant (phase-locking with event occurrences). 
        '''
        self.candidate_cycles = np.array(candidate_cycles)
        self.si_thr = si_thr
        train_ts = np.array(train_ts)

        # Get windows for training
        if window_duration is None:
            self.window_duration = train_ts[-1] - train_ts[0]
        else:
            self.window_duration = window_duration

        ts_windows_start, ts_windows_end = self._get_windowed_data(
            train_ts, window_duration=self.window_duration)
        von_mises_dist_nonsz, von_mises_dist_sz = [], []
        significant_cycles = []

        for ts_start, ts_end in zip(ts_windows_start, ts_windows_end):

            start_ind = np.where(train_ts == ts_start)[0][0]
            end_ind = np.where(train_ts == ts_end)[0][0]

            train_window_ts = train_ts[start_ind:end_ind]
            train_window_labels = train_labels[start_ind:end_ind]

            # Identify significant cycles
            sig_cycles_window = self._get_SI_significant_cycles(
                train_window_labels, train_window_ts, self.candidate_cycles)

            sig_cycles_window = sig_cycles_window[:max_n_cycles]
            significant_cycles += [np.array(sig_cycles_window)]

            # Get instantaneous phase for events and all samples for each cycle
            all_events = self._compute_instantaneuos_phases(
                train_window_ts, sig_cycles_window)

            von_mises_sz_window, von_mises_nonsz_window = [], []
            for i in range(len(sig_cycles_window)):
                initial_params = [0.0, 2.0]  # mu0, kappa
                # P(phase|X=1)
                result = minimize(
                    fun=self.von_mises_nll,
                    x0=initial_params,
                    args=(all_events[:, i], train_window_labels),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.01, 10)],
                )
                mu_sz, kappa_sz = result.x
                # P(phase|X=0)
                result = minimize(
                    fun=self.von_mises_nll,
                    x0=initial_params,
                    args=(all_events[:, i], ~train_window_labels),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.01, 10)],
                )
                mu_nonsz, kappa_nonsz = result.x

                epsilon = (2 * np.pi * self.forecast_horizon) / \
                    sig_cycles_window[i]
                von_mises_sz_window += [VonMisesDistribution(
                    mu_sz, kappa_sz, epsilon, prior=self._get_seizure_priors(train_window_labels))]
                von_mises_nonsz_window += [VonMisesDistribution(
                    mu_nonsz, kappa_nonsz, epsilon, prior=self._get_seizure_priors(~train_window_labels))]

            von_mises_dist_sz += [von_mises_sz_window]
            von_mises_dist_nonsz += [von_mises_nonsz_window]

        self.ts_windows_end = np.array(ts_windows_end)
        self.significant_cycles = significant_cycles
        self.von_mises_dist_sz = von_mises_dist_sz
        self.von_mises_dist_nonsz = von_mises_dist_nonsz

    def retrain(self, train_ts, train_labels, max_n_cycles=None):
        ''' Computes likelihoods for phase bins, according to significant cycles. Von Mises fitting using MLE with a self-defined negative log-likelihood (NLL).

        Parameters
        ---------- 
        train_ts : array-like, shape (#samples, ), dtype=int64
            Contains the unix timestamp (in seconds) of the samples.
        train_labels : array-like, shape (#samples, ), dtype=bool
            Contains the labels of each sample.
        '''
        train_ts = np.array(train_ts)
        try:
            train_ts = train_ts[np.where(
                train_ts >= self.ts_windows_end[-1])[0][0]:]
        except IndexError:  # in case there's no new data
            return None

        # Get windows for training
        ts_windows_start, ts_windows_end = self._get_windowed_data(
            train_ts, window_duration=self.window_duration)
        von_mises_dist_nonsz, von_mises_dist_sz = [], []
        significant_cycles = []

        for ts_start, ts_end in zip(ts_windows_start, ts_windows_end):

            start_ind = np.where(train_ts == ts_start)[0][0]
            end_ind = np.where(train_ts == ts_end)[0][0]

            train_window_ts = train_ts[start_ind:end_ind]
            train_window_labels = train_labels[start_ind:end_ind]

            # Identify significant cycles
            sig_cycles_window = self._get_SI_significant_cycles(
                train_window_labels, train_window_ts, self.candidate_cycles)

            sig_cycles_window = sig_cycles_window[:max_n_cycles]
            significant_cycles += [np.array(sig_cycles_window)]

            # Get instantaneous phase for events and all samples for each cycle
            all_events = self._compute_instantaneuos_phases(
                train_window_ts, sig_cycles_window)

            von_mises_sz_window, von_mises_nonsz_window = [], []
            for i in range(len(sig_cycles_window)):
                initial_params = [0.0, 2.0]  # mu0, kappa
                result = minimize(
                    fun=self.von_mises_nll,
                    x0=initial_params,
                    args=(all_events[:, i], train_window_labels),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.01, 10)],
                )
                mu_sz, kappa_sz = result.x
                # P(phase|X=0)
                result = minimize(
                    fun=self.von_mises_nll,
                    x0=initial_params,
                    args=(all_events[:, i], ~train_window_labels),
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.01, 10)],
                )
                mu_nonsz, kappa_nonsz = result.x

                epsilon = (2 * np.pi * self.forecast_horizon) / \
                    sig_cycles_window[i]
                von_mises_sz_window += [VonMisesDistribution(
                    mu_sz, kappa_sz, epsilon, prior=self._get_seizure_priors(train_window_labels))]
                von_mises_nonsz_window += [VonMisesDistribution(
                    mu_nonsz, kappa_nonsz, epsilon, prior=self._get_seizure_priors(~train_window_labels))]

            von_mises_dist_sz += [von_mises_sz_window]
            von_mises_dist_nonsz += [von_mises_nonsz_window]

        if len(von_mises_dist_sz) != 0:
            self.ts_windows_end = np.concatenate(
                (self.ts_windows_end, np.array(ts_windows_end)), axis=0)
            self.significant_cycles += significant_cycles
            self.von_mises_dist_sz += von_mises_dist_sz
            self.von_mises_dist_nonsz += von_mises_dist_nonsz

    def predict(self, test_ts):
        ''' Given samples' timestamps, computes the probability of events, according to the cycles found in train.

        Parameters
        ---------- 
        test_ts : array-like, shape (#samples, ), dtype=int64
            Contains the unix timestamp (in seconds) of the samples.

        Returns
        -------
        prob : array-like, shape (#samples, ), dtype=float64
            Constaints the estimate of likelihood of an event for each sample. 
        '''
        prob = []
        for window_indx in range(len(self.significant_cycles)):
            # Estimate likelihoods for all samples corresponding to each cycle
            phases = self._compute_instantaneuos_phases(
                test_ts, self.significant_cycles[window_indx])
            estimated_prob = []
            for i in range(len(self.significant_cycles[window_indx])):
                estimated_prob += [self._compute_vonmises_with_smoothing(
                    phases[:, i], self.von_mises_dist_sz[window_indx][i], self.von_mises_dist_nonsz[window_indx][i])]
            estimated_prob = np.array(estimated_prob)
            prob += [self._compute_combined_probability(estimated_prob)]

        time_diff = test_ts[np.newaxis, :] - self.ts_windows_end[:, np.newaxis]
        return self._weighted_average_combined_probability(np.array(prob), time_diff)

    def plot_fit_dist(self, ts, labels, window_ind, unit='days'):
        ''' Polar plot of events' phase distribution for the provided cycle. Plots both a density histogram (bin frequency divided by the bin width, so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1)) and the provided PDF function. 

        Parameters
        ---------- 
        ts : array-like, shape (#samples,), dtype=int64
            Contains the unix timestamp (in seconds) of the samples.
        labels : array-like, shape (#samples,), dtype=bool
            Contains the labels of each sample.
        window_ind : int
            Index of window from which the significant cycles should be plotted.
        '''
        if self.von_mises_dist_sz is None:
            raise ValueError(
                'Train is required before plotting the distribution.')

        von_mises_dist_sz = self.von_mises_dist_sz[window_ind]
        von_mises_dist_nonsz = self.von_mises_dist_nonsz[window_ind]

        event_bin_counts, sample_bin_counts, func_list, cycle_list, bin_edges = [], [], [], [], []
        for ind, cycle_period in enumerate(self.significant_cycles[window_ind]):

            if unit == 'days':
                cycle_list += [
                    f'{pd.to_timedelta(cycle_period, unit="s").days}-day cycle']
            elif unit == 'hours':
                cycle_list += [
                    f'{int(pd.to_timedelta(cycle_period, unit="s").total_seconds()/3600)}-hour cycle']
            else:
                raise ValueError(
                    f'Unit {unit} is not implemented. Choose between "days" and "hours".')

            def vonmises_func(x, sz_dist=von_mises_dist_sz[ind], nonsz_dist=von_mises_dist_nonsz[ind]
                              ): return self._compute_vonmises_with_smoothing(x, sz_dist, nonsz_dist)
            func_list += [vonmises_func]

            epsilon = (2 * np.pi * self.forecast_horizon) / cycle_period
            bin_edges += [np.arange(-epsilon/2, 2*np.pi, epsilon)]

            phases_events = self._compute_instantaneuos_phases(
                ts[labels == True], [cycle_period])
            phases_all = self._compute_instantaneuos_phases(ts, [cycle_period])
            event_bin_counts += [self._compute_bin_counts(
                phases_events, bin_edges[-1])]
            sample_bin_counts += [self._compute_bin_counts(
                phases_all, bin_edges[-1])]

        plot_event_phase_dist(bin_edges, event_bin_counts,
                              sample_bin_counts, func_list, cycle_list)

    def _compute_vonmises_with_smoothing(self, theta, sz_dist, nonsz_dist):
        '''Internal method that computes an estimate of the point value of the von mises distribution, through a slice of its CDF, with smoothing (to avoid zero probability estimates). The width of the slice corresponds to the duration of the forecast horizon (e.g. in a 24h cycle and forecast horizon of 1h, the width of the slice equals 2pi/24). "sz_dist" and "nonsz_dist" are instances of VonMisesDistribution.'''
        likelihood = sz_dist.get_probability(theta)
        marginal = likelihood * sz_dist.prior + \
            nonsz_dist.get_probability(theta) * nonsz_dist.prior
        # P(X=1|phase) = (P(phase|X=1) * P(X=1)) / P(phase)
        return (likelihood * sz_dist.prior) / marginal

    def von_mises_nll(self, params, theta, outcomes):
        '''negative log likelihood'''
        mu0, kappa = params
        probs = scipy.special.expit(kappa * np.cos(theta - mu0))
        nll = -np.sum(outcomes * np.log(probs))
        return nll

    def _get_seizure_priors(self, labels):
        '''Internal method that computes the seizure prior P(X=1) as the ratio of seizure samples to total samples.'''
        return np.sum(labels) / len(labels)


def exclude_harmonics(periods):
    """Remove harmonics from the list of periods."""
    # periods_sorted = sorted(periods, reverse=True)
    non_harmonics = set()

    for p1 in periods:
        # Check if p1 is harmonic of any previously added period
        is_harmonic = False
        for p2 in non_harmonics:
            if (p2 % p1 == 0 or p1 % p2 == 0):
                is_harmonic = True
                break

        if not is_harmonic:
            non_harmonics.add(p1)

    return list(non_harmonics)


class VonMisesDistribution:
    ''' Von Mises (aka circular normal distribution) distribution. 

    Attributes
    ---------- 
    mu : float
        Measure of location (the distribution is clustered around mu). Analogous to mean in the normal distribution.
    kappa : float
        Measure of concentration. Analogous to the inverse of variance in the normal distribution.
    epsilon : float
        Point estimate of probability is computed as a slice of the pdf, where the width is equal to this parameter.
    prior : float
        Prior of positive class P(X=1).
    smoothing : float
        Smoothing factor to ensure that probabilities computed from the von Mises distribution are never 0.

    Methods
    -------
    get_probability(theta) :
        Compute estimate of probability from the von Mises distribution.
    '''

    def __init__(self, mu, kappa, epsilon, prior, smoothing=1E-6):
        self.mu = mu
        self.kappa = kappa
        self.epsilon = epsilon
        self.prior = prior
        self.smoothing = smoothing

        self.n_slices = (2*np.pi) / epsilon

    def get_probability(self, theta):
        ''' Compute estimate of probability from the von Mises distribution. Computed as a slice of the pdf, where the width is equal to "epsilon".

        Parameters
        ---------- 
        theta : np.array, shape (#samples,)
            Phase of each sample with regards to a particular cycle. 

        Returns
        -------
        prob : np.array, shape (#samples,)
            Estimate of probability from the von Mises distribution.
        '''
        prob = vonmises.cdf(theta+self.epsilon/2, kappa=self.kappa, loc=self.mu) - \
            vonmises.cdf(theta-self.epsilon/2, kappa=self.kappa, loc=self.mu)
        # prob = (prob + self.smoothing)/(1+len(prob)*self.smoothing)
        return prob
