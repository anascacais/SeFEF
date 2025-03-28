# built-in
import os

# third-party
import h5py
import numpy as np
import pandas as pd

# local
from config import forecast_horizon, directory_information
from seizureforecast.optimize_threshold import optimize_thr_GMM
from seizureforecast.prepare_data import create_events_dataset
from seizureforecast.model_periodicity_analysis import VonMisesEstimator

# SeFEF
from sefef import labeling, evaluation, postprocessing, visualization, scoring

# Data preparation - read files
event_times_metadata = pd.read_csv(os.path.join(directory_information['data_folder_path'], 'event_times_metadata.csv'))
with open(os.path.join(directory_information['data_folder_path'], 'synthetic_onsets.txt'), 'r') as f:
    event_onsets = [float(line.strip()) for line in f]

create_events_dataset(event_onsets, freq=['D', 'h'][forecast_horizon < 60*60*24], dataset_filepath=os.path.join(
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
    dataset = evaluation.Dataset(timestamps=event_times_dataset['timestamps'][(
    )], samples_duration=[forecast_horizon]*len(event_times_dataset['timestamps'][(
    )]), sz_onsets=event_times_dataset['sz_onsets'][()])
    tscv.split(dataset)
    tscv.plot(dataset)

    # Operationalizing CV
    figs = []
    for ifold, (train_data, test_data) in enumerate(tscv.iterate(event_times_dataset)):
        print(
            f'\n---------------------\nStarting TSCV fold {ifold+1}/{tscv.n_folds}\n---------------------')

        X_train, y_train, ts_train, sz_onsets_train = train_data
        X_test, _, ts_test, sz_onsets_test = test_data

        seizure_hist_freq = pd.to_datetime(pd.Series(sz_onsets_train), unit='s').dt.floor(['D', 'h'][forecast_horizon < 3600*24]).nunique(
        ) / pd.to_datetime(pd.Series(ts_train), unit='s').dt.floor(['D', 'h'][forecast_horizon < 3600*24]).nunique()
        print(f'Historical seizure frequency: {seizure_hist_freq}')

        # List underlying cycles with periods ranging from 2-periods to 60-periods
        total_duration = pd.to_timedelta(
            (ts_train[-1] - ts_train[0]) + forecast_horizon, unit='s')
        fast_cycles = [pd.Timedelta(hours=t) for t in [6, 12, 24]]
        slow_cycles = [pd.Timedelta(days=t) for t in list(
            range(3, min([60, int(np.floor(total_duration.days * 0.5)+1)])))]
        candidate_cycles = fast_cycles + slow_cycles
        candidate_cycles = [cycle for cycle in candidate_cycles if cycle > pd.to_timedelta(
            forecast_horizon, unit='s')]

        # Compute likelihoods for phase bins, according to significant cycles.
        estimator = VonMisesEstimator(forecast_horizon=forecast_horizon)
        try:
            estimator.train(train_ts=X_train, train_labels=y_train,
                            candidate_cycles=[cycle.total_seconds() for cycle in candidate_cycles], si_thr=0.6)
            estimator.plot_fit_dist(X_train, y_train)
        except ValueError as e:
            print(e)
            continue

        #  Optimize high-probability threshold
        high_likelihood_thr = optimize_thr_GMM(np.reshape(
            estimator.predict(test_ts=X_train), (-1, 1)))

        # Compute probability estimates given samples' timestamps
        pred = estimator.predict(test_ts=X_test)

        # SeFEF - postprocessing module
        forecast = postprocessing.Forecast(pred, ts_test)
        forecasts, ts = forecast.postprocess(
            forecast_horizon=forecast_horizon, smooth_win=2*60*60, origin='clock-time')

        # SeFEF - visualization module
        fig = visualization.plot_forecasts(
            forecasts, ts,  sz_onsets_test, high_likelihood_thr, forecast_horizon, title='Daily seizure probability', return_plot=True, show=False)
        figs += [fig]

        # SeFEF - scoring module
        scorer = scoring.Scorer(metrics2compute=['Sen', 'FPR', 'TiW', 'AUC_TiW', 'resolution', 'reliability', 'BS', 'skill'],
                                sz_onsets=sz_onsets_test,
                                forecast_horizon=forecast_horizon,
                                reference_method='prior_prob',
                                hist_prior_prob=seizure_hist_freq)

        fold_performance = scorer.compute_metrics(
            forecasts, ts, binning_method='uniform', num_bins=5, draw_diagram=True, threshold=high_likelihood_thr)

        # Print results
        for metric in fold_performance:
            fold_performance[metric] = f'{fold_performance[metric]:0.3f}'
        print(fold_performance)

    # SeFEF - visualization module
    if len(figs) != 0:
        visualization.aggregate_plots(figs)

except KeyboardInterrupt:
    print('Interrupted by user.')
except Exception as e:
    print(e)
finally:
    event_times_dataset.close()
