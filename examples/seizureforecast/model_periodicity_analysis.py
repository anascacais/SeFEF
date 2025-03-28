# third-party
import numpy as np
import pandas as pd
from scipy.stats import vonmises
import scipy
import scipy.special as sspecial
import numpy as np

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
        # Get magnitude of mean vector (aka SI, aka PLV) with penalization factor
        si = np.abs(np.mean(np.exp(1j * phases), axis=0)) - 1/len(phases)

        # Indentify significant cycles
        si_indx = scipy.signal.find_peaks(np.array([0, *si, 0]))[0] - 1
        candidate_cycles = candidate_cycles[si_indx]
        si = si[si_indx]
        significant_cycles = candidate_cycles[np.where(si >= self.si_thr)]

        significant_cycles = [val for _, val
                              in sorted(zip(si[si >= self.si_thr], significant_cycles), reverse=True)]
        significant_cycles = exclude_harmonics(significant_cycles)

        if len(significant_cycles) == 0:
            raise ValueError('No significant cycles were found.')

        return significant_cycles

    def _validate_candidate_cycles(self, ts, candidate_cycles):
        '''Internal method that returns only the candidate cycles that are smaller than 1/2 of total duration of data.'''
        total_duration = ts[-1] - ts[0]
        return candidate_cycles[candidate_cycles < 0.5 * total_duration]

    def _compute_combined_probability(self, cycle_based_odds, sz_prior):
        '''Internal method that computed the combined probability resulting from multiple cycles, through [TBD]. Input has shape (#cycles, #samples).'''
        return ((np.prod(cycle_based_odds, axis=0)**(1/cycle_based_odds.shape[0])) * sz_prior)


class VonMisesEstimator(EventProbabilityEstimator):
    ''' Estimator of event probability based on a von Mises distribution that is fitted to the train data.

    Attributes
    ---------- 
    forecast_horizon : int
        Forecast horizon in seconds.  
    significant_cycles : array, shape (#cycles,), dtype=int64
        Array of periods (in seconds) of significant cycles. 
    candidate_cycles : array-like, shape (#cycles,), dtype=int64
        Contains periods (in seconds) of candidate cycles.
    si_thr : float
        Minimum value of Synchronization Index (SI) to consider a cycle as significant (phase-locking with event occurrences).
    von_mises_dist_sz : list of VonMisesDistribution instances
        VonMisesDistribution instances for seizure events. Each element of the list corresponds to a significant cycle.
    von_mises_dist_sz : list of VonMisesDistribution instances
        VonMisesDistribution instances for non-seizure events. Each element of the list corresponds to a significant cycle.

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

    def train(self, train_ts, train_labels, candidate_cycles, si_thr, max_n_cycles=None, **args):
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
        train_labels = train_labels.astype(bool)

        von_mises_dist_sz, von_mises_dist_nonsz, significant_cycles = self._get_VonMisesDist(
            train_ts, train_labels, max_n_cycles)

        self.sz_prior = self._get_seizure_priors(train_labels)
        self.nonsz_prior = self._get_seizure_priors(~train_labels)

        self.significant_cycles = np.array(significant_cycles)
        self.von_mises_dist_sz = von_mises_dist_sz
        self.von_mises_dist_nonsz = von_mises_dist_nonsz

    def _get_VonMisesDist(self, train_ts, train_labels, max_n_cycles):
        '''Internal method that computes an estimate of VonMises PMF distribution for the ictal (X=1) and interictal samples (X=0).'''
        # Identify significant cycles
        sig_cycles = self._get_SI_significant_cycles(
            train_labels, train_ts, self.candidate_cycles)

        sig_cycles = sig_cycles[:max_n_cycles]

        # Get instantaneous phase for events and all samples for each cycle
        all_events = self._compute_instantaneuos_phases(
            train_ts, sig_cycles)

        von_mises_sz, von_mises_nonsz = [], []
        for i in range(len(sig_cycles)):
            if pd.Timedelta(seconds=sig_cycles[i]).days > 1:
                unit = 24*60*60
            else:
                unit = 60*60
            epsilon = (2 * np.pi * unit) / sig_cycles[i]
            # P(phase|X=1)
            dist_sz = VonMisesDistribution(epsilon)
            dist_sz._estimate_parameters(all_events[:, i][train_labels])

            # P(phase|X=0)
            dist_nonsz = VonMisesDistribution(epsilon)
            dist_nonsz._estimate_parameters(all_events[:, i][~train_labels])

            von_mises_sz += [dist_sz]
            von_mises_nonsz += [dist_nonsz]

        return von_mises_sz, von_mises_nonsz, sig_cycles

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
        # Estimate likelihoods for all samples corresponding to each cycle
        phases = self._compute_instantaneuos_phases(
            test_ts, self.significant_cycles)

        estimated_odds = []
        for i in range(len(self.significant_cycles)):
            estimated_odds += [self._compute_vonmises_odds(
                phases[:, i], self.von_mises_dist_sz[i], self.von_mises_dist_nonsz[i])]

        estimated_odds = np.array(estimated_odds)
        prob = self._compute_combined_probability(
            estimated_odds, self.sz_prior)

        return np.array(prob)

    def plot_fit_dist(self, ts, labels):
        ''' Polar plot of events' phase distribution for the provided cycle. Plots both a density histogram (bin frequency divided by the bin width, so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1)) and the provided PDF function. 

        Parameters
        ---------- 
        ts : array-like, shape (#samples,), dtype=int64
            Contains the unix timestamp (in seconds) of the samples.
        labels : array-like, shape (#samples,), dtype=bool
            Contains the labels of each sample.
        '''
        if self.von_mises_dist_sz is None:
            raise ValueError(
                'Train is required before plotting the distribution.')

        event_bin_counts, sample_bin_counts, func_list, cycle_list, bin_edges = [], [], [], [], []
        for ind, cycle_period in enumerate(self.significant_cycles):
            if pd.Timedelta(seconds=cycle_period).days > 1:
                unit = 24*60*60
                cycle_list += [
                    f'{pd.to_timedelta(cycle_period, unit="s").days}-day cycle']
            else:
                unit = 60*60
                cycle_list += [
                    f'{int(pd.to_timedelta(cycle_period, unit="s").total_seconds()/3600)}-hour cycle']

            def vonmises_func(x, sz_dist=self.von_mises_dist_sz[ind], nonsz_dist=self.von_mises_dist_nonsz[ind]
                              ): return self._compute_vonmises_prob(x, sz_dist, nonsz_dist)
            func_list += [vonmises_func]

            epsilon = (2 * np.pi * unit) / cycle_period
            bin_edges += [np.arange(0, 2*np.pi+epsilon/2, epsilon)]

            phases_events = self._compute_instantaneuos_phases(
                ts[labels == True], [cycle_period])
            phases_all = self._compute_instantaneuos_phases(ts, [cycle_period])
            event_bin_counts += [self._compute_bin_counts(
                phases_events, bin_edges[-1])]
            sample_bin_counts += [self._compute_bin_counts(
                phases_all, bin_edges[-1])]

        plot_event_phase_dist(bin_edges, event_bin_counts,
                              sample_bin_counts, func_list, cycle_list)

    def _compute_vonmises_odds(self, theta, sz_dist, nonsz_dist):
        '''Internal method that computes an estimate of the point value of the von mises distribution, through a slice of its CDF, with smoothing (to avoid zero probability estimates). The width of the slice corresponds to the duration of the forecast horizon (e.g. in a 24h cycle and forecast horizon of 1h, the width of the slice equals 2pi/24). "sz_dist" and "nonsz_dist" are instances of VonMisesDistribution.'''
        likelihood = sz_dist.get_probability(theta)
        marginal = likelihood * self.sz_prior + \
            nonsz_dist.get_probability(theta) * self.nonsz_prior
        # P(X=1|phase) = P(phase|X=1) / P(phase)
        return likelihood / marginal

    def _compute_vonmises_prob(self, theta, sz_dist, nonsz_dist):
        return (self._compute_vonmises_odds(theta, sz_dist, nonsz_dist) * self.sz_prior)

    def _get_seizure_priors(self, labels):
        '''Internal method that computes the seizure prior P(X=1) as the ratio of seizure samples to total samples.'''
        return np.sum(labels) / len(labels)


def exclude_harmonics(periods):
    """Remove harmonics from the list of periods."""
    # periods_sorted = sorted(periods, reverse=True)
    non_harmonics_slow = set()
    non_harmonics_fast = set()

    slow_periods = np.array(periods)[np.where(
        np.array([pd.Timedelta(seconds=p).days for p in periods]) > 1)]
    fast_periods = np.array(periods)[np.where(
        ~(np.array([pd.Timedelta(seconds=p).days for p in periods]) > 1))]

    for p1 in slow_periods:
        # Check if p1 is harmonic of any previously added period
        is_harmonic = False
        for p2 in non_harmonics_slow:
            if (p2 % p1 == 0 or p1 % p2 == 0):
                is_harmonic = True
                break

        if not is_harmonic:
            non_harmonics_slow.add(p1)

    for p1 in fast_periods:
        # Check if p1 is harmonic of any previously added period
        is_harmonic = False
        for p2 in non_harmonics_fast:
            if (p2 % p1 == 0 or p1 % p2 == 0):
                is_harmonic = True
                break

        if not is_harmonic:
            non_harmonics_fast.add(p1)

    return list(non_harmonics_slow) + list(non_harmonics_fast)


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

    def __init__(self, epsilon, smoothing=1E-6):
        self.mu = None
        self.kappa = None
        self.pmf = None

        self.smoothing = smoothing

        self.epsilon = epsilon
        self.bin_edges = np.linspace(
            0, 2*np.pi, np.ceil((2*np.pi) / epsilon).astype('int64') + 1)

    def _compute_PMF(self):
        '''Internal method that computes an approximation of the von Mises probability mass function (PMF), assuming a number of discrete elements equivalent to dividing the unit circle into slices with width "epsilon".'''
        pmf = vonmises.cdf(self.bin_edges[1:], kappa=self.kappa, loc=self.mu) - \
            vonmises.cdf(
                self.bin_edges[:-1], kappa=self.kappa, loc=self.mu) + np.finfo(np.float64).eps
        # normalization due to floating-point precision limitations
        pmf /= np.sum(pmf)
        return pmf

    def _estimate_parameters(self, phases, num_iter=5):
        """
        vonMisesEstimation
        ------------------
        von Mises-Fischer parameter estimation methods for 2-D case
        (proper von Mises distribution).

        :copyright: Cognitive Systems Lab, 2025

        Estimate von Mises parameters `mu` as the angle of the mean complex phasor, and `k` using a modified version of Sra's truncated
        Newton approximation, where the number of iterations is increased to improve accuracy in the 2-Dimensional case.

        Parameters
        ----------
        phases : arraylike
            Array containing observed phases as radians between $-\pi$ and $\pi$.
        num_iter : int, optional
            Number of iterations to use in Newton's approximation method, by default 5

        Returns
        -------
        mu : float
            mu parameter
        k : float
            k parameter
        """
        complex_phases = np.exp((1.0j) * phases)
        mean_phase = np.mean(complex_phases)
        R = np.abs(mean_phase)

        mu = np.angle(mean_phase)

        Rho = R - 1/len(phases)

        if (1 - Rho) < 1E-12:
            self.mu, self.kappa = mu, 1E12

        elif (Rho < 0):
            self.mu, self.kappa = mu, 1E-12
            self.pmf = self._compute_PMF()

        else:
            k = Rho * (1 - Rho**2) / (1 - Rho)

            def A2(k):
                return sspecial.i1(k) / sspecial.i0(k)

            for _ in range(num_iter):
                k -= (A2(k) - Rho) / (1 - A2(k)**2 - (A2(k) / k))

            self.mu, self.kappa = mu, k

        self.pmf = self._compute_PMF()

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
        bins = np.digitize(theta, self.bin_edges)-1
        prob = self.pmf[bins]
        return prob
