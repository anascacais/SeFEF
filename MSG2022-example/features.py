
# third-party
import numpy as np
import scipy

def extract_features(samples, channels_names, features2extract, sampling_frequency):
    ''' Extract features from "features2extract".  
    
    Parameters
    ---------- 
    samples: array-like, shape (#samples, #data points in sample, #channels)
        Data array.
    timestamps: array-like, shape (#samples, )
        Contains the start timestamp of each sample.
    channels_names: list<str>
        List of strings, corresponding to the names of the channels.
    features2extract: dict
        Dict where keys correspond to the names of the channels in "channels_names" and the values are lists with the names of features. The features can be statistical, EDA event-based, or Hjorth features. Feature names should be the ones from the following list:
            - Statistical: "mean", "power", "std", "kurtosis", "skewness", "mean_1stdiff", "mean_2nddiff", "entropy".
            - EDA event-based: "SCR_amplitude", "SCR_peakcount", "mean_SCR_amplitude", "mean_SCR_risetime", "sum_SCR_amplitudes", "sum_SCR_risetimes", "SCR_AUC".
            - Hjorth: "hjorth_activity", "hjorth_mobility", "hjorth_complexity".
    sampling_frequency: int
        Frequency at which the data is presented. 
        
    Returns
    -------
    features: array-like, shape (#samples, #data points in sample)
       Data array with features extracted.

    Raises
    ------
    ValueError :
        Raised when a feature name in "features2extract" is not a valid feature. 
    ''' 
    if samples is None:
        return None
    
    features = []

    feat2function = {'mean': _extract_mean, 'power': _extract_power, 'std': _extract_std, 'kurtosis': _extract_kurtosis, 'skewness': _extract_skewness,
                    'mean_1stdiff': _extract_mean_1stdiff, 'mean_2nddiff': _extract_mean_2nddiff, 'shannon_entropy': _extract_shannon_entropy, 'SCR_amplitude': _extract_SCR_amplitude, 'SCR_peakcount': _extract_SCR_peakcount, 'mean_SCR_amplitude': _extract_mean_SCR_amplitude, 'mean_SCR_risetime': _extract_mean_SCR_risetime, 'sum_SCR_amplitudes': _extract_sum_SCR_amplitudes, 'sum_SCR_risetimes': _extract_sum_SCR_risetimes, 'SCR_AUC': _extract_SCR_AUC, 'hjorth_activity': _extract_hjorth_activity, 'hjorth_mobility': _extract_hjorth_mobility, 'hjorth_complexity': _extract_hjorth_complexity}
    
    try:
        for channel in features2extract.keys():
            channel_ind = channels_names.index(channel)
            
            if any(['SCR' in ft for ft in features2extract[channel]]): 
                scr_events = _eda_events(samples[:, :, channel_ind], sampling_frequency)

            for feature_name in features2extract[channel]:
                if feature_name not in feat2function.keys():
                    raise ValueError(f"{feature_name} is not a valid feature.")
                
                if 'SCR' in feature_name:
                    new_feature = feat2function[feature_name](scr_events)
                else:
                    new_feature = feat2function[feature_name](samples[:, :, channel_ind])
                features += [new_feature]
        
        return np.concatenate(features, axis=1)
    
    except RuntimeError as e:
        print(e)
        return None



# Statistical features
def _extract_mean(array):
    """Internal method that computes the mean of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return np.mean(array, axis=1)[:, np.newaxis]

def _extract_power(array):
    """Internal method that computes the average power of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return np.mean(array**2, axis=1)[:, np.newaxis]

def _extract_std(array):
    """Internal method that computes the standard deviation of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return np.std(array, axis=1)[:, np.newaxis]

def _extract_kurtosis(array):
    """Internal method that computes the kurtosis of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return scipy.stats.kurtosis(array)[:, np.newaxis]

def _extract_skewness(array):
    """Internal method that computes the skewness of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return scipy.stats.skew(array)[:, np.newaxis]

def _extract_mean_1stdiff(array):
    """Internal method that computes the mean of the first difference of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return np.mean(np.diff(array, axis=1), axis=1)[:, np.newaxis]

def _extract_mean_2nddiff(array):
    """Internal method that computes the mean of the second difference of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return np.mean(np.diff(np.diff(array, axis=1), axis=1), axis=1)[:, np.newaxis]

def _extract_shannon_entropy(array):
    """Internal method that computes the Shannon entropy of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1)."""
    return scipy.stats.entropy(array, axis=1)[:, np.newaxis]


# EDA event-based features
def _extract_SCR_amplitude(scr_events):
    """Internal method that computes the ???? from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    raise NotImplementedError

def _extract_SCR_peakcount(scr_events):
    """Internal method that computes the number of peaks from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    return np.array([len(s) for s in scr_events[0]])[:, np.newaxis]

def _extract_mean_SCR_amplitude(scr_events):
    """Internal method that computes the mean SCR amplitude from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    return np.array([np.mean(s) for s in scr_events[0]])[:, np.newaxis]

def _extract_mean_SCR_risetime(scr_events):
    """Internal method that computes the mean SCR rise time (in seconds) from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    return np.array([np.mean(s) for s in scr_events[1]])[:, np.newaxis]

def _extract_sum_SCR_amplitudes(scr_events):
    """Internal method that computes the sum of SCR amplitudes from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    return np.array([np.sum(s) for s in scr_events[0]])[:, np.newaxis]

def _extract_sum_SCR_risetimes(scr_events):
    """Internal method that computes the sum of SCR rise times (in seconds) from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    return np.array([np.sum(s) for s in scr_events[1]])[:, np.newaxis]

def _extract_SCR_AUC(scr_events):
    """Internal method that computes ??? from the tuple 'scr_events' (which contains 'amps' and 'rise_times'), and returns an array with shape (#samples, 1)."""
    raise NotImplementedError


# Hjorth features
def _extract_hjorth_activity(array):
    """Internal method that computes the Hjorth activity of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1). Implemented as described in Shukla et al. (2019), IEEE Transactions on Affective Computing."""
    return np.sum((array - np.mean(array, axis=1)[:, np.newaxis])**2, axis=1)[:, np.newaxis]

def _extract_hjorth_mobility(array):
    """Internal method that computes the Hjorth mobility of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1). Implemented as described in Shukla et al. (2019), IEEE Transactions on Affective Computing."""
    return np.sqrt(np.multiply(np.var(array, axis=1), np.var(np.diff(array, axis=1), axis=1)))[:, np.newaxis]

def _extract_hjorth_complexity(array):
    """Internal method that computes the Hjorth complexity of the samples in an array with shape (#samples, #data points in sample), and returns an array with shape (#samples, 1). Implemented as described in Shukla et al. (2019), IEEE Transactions on Affective Computing."""
    return np.multiply(_extract_hjorth_mobility(array), _extract_hjorth_mobility(np.diff(array, axis=1)))[:, np.newaxis]





def _eda_events(array, sampling_frequency, min_amplitude=0.01):
    """Internal method that identifies EDA events and extracts the corresponding onsets (tuple that indexes the onsets through 'array[onsets]'), peaks (tuple that indexes the peaks through 'array[peaks]'), amplitudes and rise times (both lists of N (#samples) arrays, each containing the amplitudes of the EDA events in the corresponding sample)."""

    onsets_indx = _find_extrema_indx(array, mode='min')  # get zeros
    peaks_indx = _find_extrema_indx(array, mode='max')
    
    try:
        onsets_indx, peaks_indx = _remove_unwanted_extrema(onsets_indx, peaks_indx)
    except RuntimeError:
        raise RuntimeError('No extrema found in any of the samples.')

    if len(onsets_indx) != len(peaks_indx):
        raise RuntimeError("Found incomplete EDA events")

    # get all amplitudes
    all_amps = array[(peaks_indx[:,0], peaks_indx[:,1])] - array[(onsets_indx[:,0], onsets_indx[:,1])]

    # get amplitudes that respect 'min_amplitude'
    valid_event_indx = np.argwhere(all_amps >= min_amplitude).flatten()
    
    if len(valid_event_indx) != 0:
        valid_event_sample_indx = onsets_indx[valid_event_indx][:,0]
        sample_indx_split = np.argwhere(np.concatenate(([False], np.diff(valid_event_sample_indx).astype('bool')))).flatten()

        # get onsets, peaks, amplitudes, and rise times
        onsets = (onsets_indx[valid_event_indx][:, 0], onsets_indx[valid_event_indx][:, 1])
        peaks = (peaks_indx[valid_event_indx][:, 0], peaks_indx[valid_event_indx][:, 1])
        amps = np.split(all_amps[all_amps >= min_amplitude], sample_indx_split)
        rise_times = np.split((peaks[1] - onsets[1]) / sampling_frequency, sample_indx_split)

        samples_indx_no_events = list(set(range(len(array))) - set(valid_event_sample_indx)) 
        for s in samples_indx_no_events:
            amps.insert(s, np.array([]))
            rise_times.insert(s, np.array([]))
    else:
        raise RuntimeError('No extrema found in any of the samples.')

    return (amps, rise_times)
    

def _remove_unwanted_extrema(onsets_indx, peaks_indx):
    """Internal method that received a set of extrema indices (corresponding to an array with shape (#samples, #points in sample)) and removes the first peaks if they are not preceeded by an onset and removes the last onsets if they are not followed by a peak."""

    onsets_indx_by_sample = np.split(onsets_indx, np.argwhere(np.concatenate(([False], np.diff(onsets_indx[:,0]).astype('bool')))).flatten())
    peaks_indx_by_sample = np.split(peaks_indx, np.argwhere(np.concatenate(([False], np.diff(peaks_indx[:,0]).astype('bool')))).flatten())

    if (len(onsets_indx_by_sample[0]) == 0 or len(peaks_indx_by_sample[0]) == 0):
        raise RuntimeError('No extrema found in any of the samples.')
    
    # remove first peak if before onset and last onset if after peak
    for i in range(len(peaks_indx_by_sample)):
        peaks_indx_by_sample[i] = peaks_indx_by_sample[i][(peaks_indx_by_sample[i][0][1] < onsets_indx_by_sample[i][0][1]):]
        onsets_indx_by_sample[i] = onsets_indx_by_sample[i][:(len(onsets_indx_by_sample[i]) - (onsets_indx_by_sample[i][-1][1] > peaks_indx_by_sample[i][-1][1]))]

    onsets_indx = np.concatenate(onsets_indx_by_sample)
    peaks_indx = np.concatenate(peaks_indx_by_sample)

    return onsets_indx, peaks_indx

def _find_extrema_indx(array=None, mode="both"):
    """Locate local extrema points in a signal, returning an array of the indices of the extrema, shape (N, array.ndim), where N is the number of extrema. Adapted from BioSSPy. Based on Fermat's Theorem."""

    # check inputs
    if array is None:
        raise TypeError("Please specify an input signal.")

    if mode not in ["max", "min", "both"]:
        raise ValueError("Unknwon mode %r." % mode)

    aux = np.diff(np.sign(np.diff(array, axis=1)), axis=1)

    if mode == "both":
        aux = np.abs(aux)
        inflection_points = aux > 0
    elif mode == "max":
        inflection_points = aux < 0
    elif mode == "min":
        inflection_points = aux > 0
    
    extrema = np.zeros_like(array, dtype=bool)
    extrema[:, 1:-1] = inflection_points[:, :]

    return np.argwhere(extrema)


# Others 

def extract_ts_features(start_timestamps, samples, channels_names, features_list):
    ''' Compute features from features_list and keep original time series as channels. 
    
    Parameters
    ---------- 
    raw_np_timestamps: array-like, shape (# samples, )
        Contains the start timestamp of each sample.
    raw_np_data: array-like, shape (# samples, # data points in sample, # channels)
        Data array.
    channels_names: list<str>
        List of strings, corresponding to the names of the channels.
    features_list: list
        List of strings containing the names of features do extract. Options: ToD (computes time of day as the hour-part of the 24h); FFT (computes Fourier Transform for BVP, EDA, TEMP, HR, and ACCMag); SQI [ACCMag, EDA, BVP] (computes signal quality index for ACCMag, EDA or BVP).

    Returns
    -------
    list_data: array-like, shape (# samples, # data points in sample, # channels)
       Data array where channels correspond to the original channels plus the new extracted features.
    ''' 

    if samples is not None:
        data = np.stack(samples) # (#samples, sample length, #channels)

        for feature in features_list:
            if feature == 'ToD':
                new_features = compute_ToD(start_timestamps, sample_length=data.shape[1])

            # elif feature == 'FFT':
            #     channel_idx = [channels_names.index(chn) for chn in ['acc_mag', 'bvp', 'eda', 'hr', 'temp']]
            #     new_features = compute_fft(data[:, :, channel_idx])

            elif feature == 'SQI ACCMag':
                channel_idx = [channels_names.index(chn) for chn in ['acc_mag']]
                new_features = compute_sqi_acc(data[:, :, channel_idx])

            elif feature == 'SQI EDA':
                channel_idx = [channels_names.index(chn) for chn in ['eda']]
                new_features = compute_sqi_eda(data[:, :, channel_idx])

            elif feature == 'SQI BVP':
                channel_idx = [channels_names.index(chn) for chn in ['bvp']]
                new_features = compute_sqi_bvp(data[:, :, channel_idx])
            
            else:
                ValueError()

            data = np.concatenate((data, new_features), axis=2)
        
        # # transform first dimension (samples) into list
        # list_data = np.split(data, data.shape[0], axis=0)
        # list_data = [np.squeeze(x, axis=0) for x in list_data]
        
        return data
        


def compute_ToD(start_timestamps, sample_length, fs=16):
    ''' Return array with dimension (#samples, sample_length, 1) with the corresponding hour-part of the 24h.
    
    Parameters
    ---------- 
    start_timestamps: list<float>
        List (with length # samples) of the start timestamp of each sample.
    sample_length: int
        Length of the samples. Needed to expand start_timestamps into a timestamp for each point in the sample.
    
    Returns
    -------
    result: np.array
        Array with dimension (#samples, sample_length, 1) with the corresponding hour-part of the 24h.
    ''' 

    start_timestamps = np.array(start_timestamps)
    time_offsets = np.arange(sample_length) * (1 / fs)
    timestamps = start_timestamps[:, np.newaxis] + time_offsets

    datetimes = timestamps.astype('datetime64[s]')
    tod = datetimes.astype('datetime64[h]').astype(int) % 24

    return tod[:, :, np.newaxis]



def compute_fft(data):
    ''' Compute Fourier transform of all axis in data. NOT WELL COMPUTED
    
    Parameters
    ---------- 
    data: np.array
        Array with dimension (#samples, sample length, #channels)
    
    Returns
    -------
    result: np.array
        Array with dimension (#samples, sample length, #channels), each channel corresponding to the FFT of each channel in the original data.
    ''' 

    return scipy.fft.fft(data, axis=1)



def compute_sqi_acc(data, segment_duration=4, fs=16): # segment length: 128Hz * 4s
    ''' Compute Signal Quality Index for ACC magnitude according to ????, calculated over 4s and averaged across the full 1-min samples.
    
    Parameters
    ---------- 
    data: np.array
        Array with dimension (#samples, sample length, 1)
    segment_duration: int
        Duration of segment (in seconds) for which the power ratio is computed (and then averaged across all segments).
    fs: int
        Sampling frequency in Hz
    
    Returns
    -------
    result: np.array
        Array with dimension (#samples, sample length, 1), corresponding to the SQI of the original channel. In practice, a single value is computed for each sample, and expanded to the length of the samples.
    ''' 
    segment_length = segment_duration * 16
    
    data_aux = np.reshape(data, (data.shape[0], data.shape[1],))
    segments = np.stack(np.split(data_aux, np.arange(segment_length, data.shape[1], segment_length), axis=1), axis=-1) # (#samples, sample length, #segments)
    
    fft = scipy.fft.rfft(segments, axis=1)
    fft_freqs = scipy.fft.rfftfreq(segments.shape[1], d=1/fs)
    psd = np.abs(fft) ** 2

    psd[psd == 0.] = np.amax(psd)*10**-6 # to not have zero values for power ratio calculation

    narrow_band_idx = np.argwhere((fft_freqs >= 0.8) & (fft_freqs <= 5))
    broad_band_idx = np.argwhere(fft_freqs >= 0.8)

    narrow_band_power = np.sum(psd[:, narrow_band_idx, :], axis=1)
    broad_band_power = np.sum(psd[:, broad_band_idx, :], axis=1)

    power_ratio = np.mean(narrow_band_power / broad_band_power, axis=-1)

    return np.multiply(np.ones(data.shape), power_ratio[:, : , np.newaxis])



def compute_sqi_eda(data, segment_duration=1, fs=16): # segment_length: 128Hz * 1s
    ''' Compute Signal Quality Index for EDA according to ????
    
    Parameters
    ---------- 
    data: np.array
        Array with dimension (#samples, sample length, 1)
    segment_duration: int
        Duration of segments (in seconds) for which the rate of amplitude change (RAC) is computed.
    fs: int
        Sampling frequency in Hz
    
    Returns
    -------
    result: np.array
        Array with dimension (#samples, sample length, 1), corresponding to the SQI of the original channel. SQI is calculated for segments within the signal and expanded to fit the fs of the samples.  
    ''' 
    segment_length = segment_duration * fs

    data_aux = np.reshape(data, (data.shape[0], data.shape[1],))
    segments = np.stack(np.split(data_aux, np.arange(segment_length, data.shape[1], segment_length), axis=1), axis=-1) # (#samples, sample length, #segments)

    max_vals = np.amax(segments, axis=1)
    min_vals = np.amin(segments, axis=1)

    # get the maximum or minimum value, whichever comes first
    max_mask = (segments == max_vals[:, np.newaxis, :])
    min_mask = (segments == min_vals[:, np.newaxis, :])
    max_indices = np.argmax(max_mask, axis=1)
    min_indices = np.argmax(min_mask, axis=1)
    indx_relative_to = np.where(max_indices < min_indices, max_indices, min_indices)

    vals_relative_to = segments[np.arange(segments.shape[0])[:, np.newaxis], indx_relative_to, np.arange(segments.shape[2])]
    vals_relative_to[vals_relative_to == 0.] = 0.0001

    amplitudes = max_vals - min_vals
    rac = np.abs(amplitudes / vals_relative_to)
    rac = np.repeat(rac, segment_length, axis=1)

    mean_vals = np.mean(segments, axis=1)
    mean_vals = np.repeat(mean_vals, segment_length, axis=1)

    # impose conditions for EDA quality 
    sqi = np.ones_like(data_aux)
    sqi[(mean_vals < 0.05) | (rac > 0.2)] = 0.

    return sqi[:, :, np.newaxis]

# # s=1
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=data_aux[0,:], mode='lines'))
# fig.add_trace(go.Scatter(y=sqi[0,:], mode='lines'))
# #fig.add_hline(y=rac[0,s])
# fig.show()

def compute_sqi_bvp(data, segment_duration=4, fs=16):
    ''' Compute Signal Quality Index for BVP according to ????
    
    Parameters
    ---------- 
    data: np.array
        Array with dimension (#samples, sample length, 1)
    segment_duration: int
        Duration of segment (in seconds) for which the spectral entropy is computed (and then averaged across all segments).
    fs: int
        Sampling frequency in Hz
    
    Returns
    -------
    result: np.array
        Array with dimension (#samples, sample length, 1), corresponding to the SQI of the original channel. In practice, a single value of Spectral Entropy is computed for each sample (as the average of 4s-segments), and expanded to the length of the samples.
    ''' 
    segment_length = segment_duration * fs
    data_aux = np.reshape(data, (data.shape[0], data.shape[1],))
    segments = np.stack(np.split(data_aux, np.arange(segment_length, data.shape[1], segment_length), axis=1), axis=-1) # (#samples, sample length, #segments)

    fft = scipy.fft.rfft(segments, axis=1)
    psd = np.abs(fft) ** 2

    psd[psd == 0.] = np.amax(psd)*10**-6 # to not have zero values for entropy calculation
    psd_norm = psd / psd.sum()

    entropy = np.mean(-np.sum(psd_norm * np.log2(psd_norm), axis=1), axis=1) 

    # impose conditions for EDA quality 
    sqi = np.ones(entropy.shape)
    sqi[entropy >= 0.8] = 0.

    return np.multiply(np.ones(data.shape), sqi[:, np.newaxis, np.newaxis])




def remove_bad_quality_samples(start_timestamps, samples, channels_names): 

    if samples is not None:
        data = np.stack(samples) # (#samples, sample length, #channels)

        sqi_idx = [channels_names.index(chn) for chn in channels_names if 'SQI' in chn]
        good_idx = data[:, :, sqi_idx].all(axis=2).sum(axis=1) / data.shape[1] >= 0.8
       
        data = data[good_idx, :, :]
        start_timestamps = [item for item, include in zip(start_timestamps, good_idx) if include]

        try:
            list_data = np.split(data, data.shape[0], axis=0)
            list_data = [np.squeeze(x, axis=0) for x in list_data]
        except:
            list_data, start_timestamps = None, None
        
        return start_timestamps, list_data

    

