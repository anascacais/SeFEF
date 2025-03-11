from datetime import datetime

forecast_horizon = 24*60*60  # 24h in seconds
high_likelihood_thr = 0.2

directory_information = {
    'data_folder_path': 'examples/data',
    'results_folder_path': 'notebooks/working_with_timestamps_synthetic',
    'preprocessed_data_path': 'examples/preprocessed_data',
}

model_card = {
    "details": {
        "name": f"{int(forecast_horizon)}h_plv-0.8_von-mises-dist_slice-fh_windowed",
        "date": datetime.today().strftime('%Y-%m-%d'),
        "forecast_horizon": forecast_horizon,
        "description": f"Significant cycles found with PLV, threshold 0.8, harmonics removed. Von Mises distribution fit to the train event data using a self-defined NLL. Smoothing of 1E-6 added to the pdf, point estimate of probability using slice from CDF centered at phase and with width corresponding to the duration of the forecast horizon. Seizure prior and marginal phase distribution also used to compute P(X=1|phase). Combined cycle probability from geometric mean of odds. Cycles computed in windows with duration equal to the train set of first CV fold, and 50\% overlap. Combined probabilities of windows with exponential decay weighted average (decay factor 1.46E-7). Retraining after every new event.",
    },
    "training": {
        "dataset": "Synthetic Dataset: This dataset contains synthesized event occurrence timestamps spanning 2.5 years, starting from January 1, 2020. Events occur periodically, with an initial cycle of 28 days (in seconds), subject to a small random variation of ±1 day.",
        "preprocessing": "None",
    },
    "evaluation": {
        "dataset": "Synthetic Dataset: This dataset contains synthesized event occurrence timestamps spanning 2.5 years, starting from January 1, 2020. Events occur periodically, with an initial cycle of 28 days (in seconds), subject to a small random variation of ±1 day.",
        "preprocessing": "None",
        "reference_forecast": "prior probability",
    },
    "postprocessing": {
        "smooth_win": 2*60*60,
        "forecast_horizon": 'forecast_horizon',
    },
}
