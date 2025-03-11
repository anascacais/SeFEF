# third-party
import h5py


def create_events_dataset(dataset, dataset_filepath):
    ''' Create empty hdf5 file.

    Parameters
    ---------- 
    dataset : sefef.Dataset instance
        Dataset instance containing the timestamps and onsets of events.
    dataset_filepath : str
        Complete path to the hdf5 file.

    Returns
    -------
    None
    '''
    with h5py.File(dataset_filepath, 'w') as hdf:
        hdf.create_dataset('timestamps',
                           data=dataset.metadata.index.to_numpy(), maxshape=[None,], dtype='int64')
        hdf.create_dataset(
            'data', data=dataset.metadata['sz_onset'].to_numpy(), dtype='bool')
