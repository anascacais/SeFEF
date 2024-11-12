# third-party
import pandas as pd
import numpy as np


def get_metadata(data_path):
    ''' Get metadata available on event data file (aka menstruation data).

    Parameters
    ---------- 
    data_path : str
        Path to data file.

    Returns
    -------
    event_dates : array-like, dtype=datetime64[ns]
        Array with datatimes when events happened. An event (aka menstruation) lasts more than one day.
    '''
    data_df = pd.read_json(data_path)
    data_df = data_df[data_df['type'] == 'period']
    event_dates = np.sort(pd.to_datetime(data_df['date']))
    return event_dates


def get_events_onsets(event_dates, min_onset_distance='1 days'):
    """Returns a dataframe with the event onset (aka menstruation) datetime from list of events (which last more than 1 day).

    Parameters
    ----------
    event_dates : array-like
        Array with datatimes when events happened. An event (aka menstruation) lasts more than one day.
    min_onset_distance : str, defaults to "1 days"
        Minimum distance between adjacent event dates to be considered individual onsets.

    Returns
    -------
    event_timestamps : list
        Contains the unix timestamp (in seconds) of the onsets of events.
    """
    min_onset_distance = pd.to_timedelta(min_onset_distance).to_numpy()
    event_dates_dt = pd.to_datetime(event_dates)
    datetime_diff = event_dates_dt.diff()

    onsets_index = [0] + np.where(datetime_diff > min_onset_distance)[0].tolist()
    event_timestamps = event_dates_dt[onsets_index].astype(int) // 10**9

    return event_timestamps


def create_metadata_df(events_onset, freq='D'):
    ''' Function description 

    Parameters
    ---------- 
    event_timestamps : list
        Contains the unix timestamp (in seconds) of the onsets of events.
    freq : str, defaults to "D" (1 day)
        Duration (in seconds) to attribute to each segment.

    Returns
    -------
    files_metadata : pd.DataFrame
        Dataframe containing metadata on the "fake" data files. Contains the following columns:
            - 'filepath' (str): filled with NaN
            - 'first_timestamp' (Int64): unix timestamp (in seconds) of the start of the segment.
            - 'total_duration' (Int64): duration (in seconds) of the data within the file.
    '''

    files_metadata = pd.DataFrame(index=pd.to_datetime(events_onset, unit='s'), columns=['filepath', 'total_duration'])
    files_metadata = files_metadata.asfreq(freq=freq)

    files_metadata.loc[:, 'total_duration'] = files_metadata['total_duration'].astype(
        'float').fillna(pd.to_timedelta('1'+freq) / np.timedelta64(1, 's'))

    files_metadata['first_timestamp'] = files_metadata.index.astype('datetime64[ns]').astype('int') // 10**9
    files_metadata = files_metadata.astype({'filepath': str, 'total_duration': int})

    return files_metadata.reset_index(drop=True)
