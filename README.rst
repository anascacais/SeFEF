Welcome to ``SeFEF``
======================

.. image:: https://raw.githubusercontent.com/anascacais/sefef/main/docs/logo/sefef-logo.png
    :align: center
    :alt: SeFEF logo

|

``SeFEF`` is a Seizure Forecast Evaluation Framework written in Python.
The framework standardizes the development, evaluation, and reporting of individualized algorithms for seizure likelihood forecast. 
``SeFEF`` aims to decrease development time and minimize implementation errors by automating key procedures within data preparation, training/testing, and computation of evaluation metrics. 

Highlights:
-----------

- ``evaluation`` module: implements time series cross-validation.
- ``labeling`` module: automatically labels samples according to the desired pre-ictal duration and prediction latency.
- ``postprocessing`` module: processes individual predicted probabilities into a unified forecast according to the desired forecast horizon.
- ``scoring`` module: computes both deterministic and probabilistic metrics according to the horizon of the forecast.  



Installation
------------

Installation can be easily done with ``pip``:

.. code:: bash

    $ pip install sefef

Simple Example
--------------

The code below loads the metadata from an existing dataset from the ``examples`` folder, splits creates a ``Dataset`` instance, and creates an adequate split for a time series cross-validation (``TSCV``). It also provides an example of model development and evaluation through a simple probabilistic estimator that leverages periodicity in event data. 

.. code:: python

    # built-in
    import os
    import json
    import math

    # third-party
    import h5py
    import numpy as np
    import pandas as pd

    # local
    from config import forecast_horizon, directory_information, high_likelihood_thr
    from seizureforecast.prepare_data import create_events_dataset
    from seizureforecast.model_periodicity_analysis import VonMisesEstimator

    # SeFEF
    from sefef import labeling, evaluation, postprocessing, visualization, scoring

    # Data preparation - read files
    event_times_metadata = pd.read_csv(os.path.join(directory_information['data_folder_path'], 'event_times_metadata.csv'))
    with open(os.path.join(directory_information['data_folder_path'], 'synthetic_onsets.txt'), 'r') as f:
        event_onsets = json.load(f)

    dataset = evaluation.Dataset(event_times_metadata, event_onsets)
    create_events_dataset(dataset, dataset_filepath=os.path.join(
        directory_information['preprocessed_data_path'], f'event_times_dataset.h5'))

    # SeFEF - labeling module
    with h5py.File(os.path.join(directory_information['preprocessed_data_path'], f'event_times_dataset.h5'), 'r+') as h5dataset:
        if 'annotations' not in h5dataset.keys():
            labeling.add_annotations(
                h5dataset, sz_onsets_ts=event_onsets, preictal_duration=forecast_horizon, prediction_latency=0)
        if 'sz_onsets' not in h5dataset.keys():
            labeling.add_sz_onsets(
                h5dataset, sz_onsets_ts=event_onsets)

    try:
        event_times_dataset = h5py.File(os.path.join(
            directory_information['preprocessed_data_path'], f'event_times_dataset.h5'), 'r')

        # SeFEF - evaluation module
        tscv = evaluation.TimeSeriesCV(
            preictal_duration=forecast_horizon,
            prediction_latency=0,
            post_sz_interval=1*60*60,
            pre_lead_sz_interval=4*60*60,
        )
        tscv.split(dataset)
        tscv.plot(dataset)

        # Operationalizing CV
        for ifold, (train_data, test_data) in enumerate(tscv.iterate(event_times_dataset)):
            print(
                f'\n---------------------\nStarting TSCV fold {ifold+1}/{tscv.n_folds}\n---------------------')

            _, y_train, ts_train, event_onsets_train = train_data
            _, _, ts_test, event_onsets_test = test_data

            # List underlying cycles with periods ranging from 2-periods to 60-periods
            total_duration = ((ts_train[-1] - ts_train[0]) + forecast_horizon)
            candidate_cycles = np.arange(
                2*forecast_horizon, np.min([60*forecast_horizon, math.floor((total_duration*0.5) / forecast_horizon) * forecast_horizon]), forecast_horizon)
            estimator = VonMisesEstimator(forecast_horizon=forecast_horizon)

            # Compute likelihoods for phase bins, according to significant cycles.
            try:
                estimator.train(train_ts=ts_train, train_labels=y_train,
                                candidate_cycles=candidate_cycles, si_thr=0.8, window_duration=None)
                estimator.plot_fit_dist(ts_train, y_train, window_ind=-1, unit='days')
            except ValueError as e:
                print(e)
                continue

            # Compute probability estimates given samples' timestamps
            pred = estimator.predict(test_ts=ts_test)

            # SeFEF - postprocessing module
            forecast = postprocessing.Forecast(pred, ts_test)
            forecasts, ts = forecast.postprocess(
                forecast_horizon=forecast_horizon, smooth_win=2*60*60, origin='clock-time')

            # SeFEF - visualization module
            visualization.plot_forecasts(
                forecasts, ts,  event_onsets_test, high_likelihood_thr, forecast_horizon, title=f'Daily seizure probability')

            # SeFEF - scoring module
            scorer = scoring.Scorer(metrics2compute=['Sen', 'FPR', 'TiW', 'AUC_TiW', 'resolution', 'reliability', 'BS', 'skill'],
                                    sz_onsets=event_onsets_test,
                                    forecast_horizon=forecast_horizon,
                                    reference_method='prior_prob',
                                    hist_prior_prob=pd.to_datetime(pd.Series(event_onsets_train), unit='s').dt.floor('D').nunique() / pd.to_datetime(pd.Series(ts_train), unit='s').dt.floor('D').nunique())

            fold_performance = scorer.compute_metrics(
                forecasts, ts, binning_method='quantile', num_bins=5, draw_diagram=True, threshold=high_likelihood_thr)

    except KeyboardInterrupt:
        print('Interrupted by user.')
    except Exception as e:
        print(e)
    finally:
        event_times_dataset.close()
        
