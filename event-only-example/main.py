# SeFEF
from sefef import evaluation

# local
from data import get_metadata, get_events_onsets, create_metadata_df
from config import data_path


def main(data_path=data_path):

    event_dates = get_metadata(data_path)
    events_onsets = get_events_onsets(event_dates)
    metadata = create_metadata_df(events_onsets)

    dataset = evaluation.Dataset(metadata, events_onsets, sampling_frequency=None)
    pass


if __name__ == '__main__':
    main()
