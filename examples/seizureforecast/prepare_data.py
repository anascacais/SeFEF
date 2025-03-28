# third-party
import pandas as pd
import h5py


def create_events_dataset(events_onset, freq, dataset_filepath):
    ''' Create hdf5 file with timestamps of samples and corresponding feature value (middle point of day/hour). 

    Parameters
    ---------- 
    event_timestamps : list
        Contains the unix timestamp (in seconds) of the onsets of events.
    freq : str, defaults to "D" (1 day)
        Duration (as pd.offsets) to attribute to each segment.
    dataset_filepath : str
        Complete path to the hdf5 file.

    Returns
    -------
    None
    '''

    dt = pd.to_datetime(events_onset, unit='s')
    df = pd.DataFrame(dt, columns=['dt'])
    df['day'] = dt.floor(freq)

    dt = pd.to_datetime(df.groupby(by='day').mean().dt.to_numpy())

    start_dt = dt.min().floor(freq)
    end_dt = dt.max().ceil(freq)

    index = pd.date_range(start_dt - pd.Timedelta('1'+freq), end_dt, freq=freq)

    if freq == 'D':
        new_index = index + pd.to_timedelta(12, unit='h')
    else:
        new_index = index.copy()

    new_series = pd.Series(new_index)
    matching_indexes = index.floor(freq).isin(dt.floor(freq))
    new_series.loc[matching_indexes] = dt.values

    timestamps = index.astype('datetime64[ns]').astype('int') // 10**9
    data = new_series.astype('datetime64[ns]').astype('int') // 10**9

    with h5py.File(dataset_filepath, 'w') as hdf:
        hdf.create_dataset('timestamps',
                           data=timestamps, maxshape=[None,], dtype='int64')
        hdf.create_dataset(
            'data', data=data, dtype='int64')
