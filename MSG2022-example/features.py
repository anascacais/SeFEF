
# third-party
import numpy as np
import scipy

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

    

