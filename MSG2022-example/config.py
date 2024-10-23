# Store useful variables and configuration

patient_id = '2002'

data_folder_path = '/media/MSG/MSG2022/train'

train_configurations = {
    "batch_size": 32, 
    "max_epochs": 10, 
    "criterion": "BCEWithLogitsLoss", 
    "hidden_size": 128, 
    "output_size": 1, 
    "num_layers": 4, 
    "dropout": 0.2, 
    "learning_rate": 0.001, 
    "num_workers": 7,
    "data_augmentation": False,
    "val_size": 0.2,
    "channels": ["bvp", "eda", "hr", "temp", "ToD"], # channels to use
    "experiment_id": "0",
}

features2extract = {
    'acc_mag': ['mean', 'power', 'std', 'kurtosis', 'skewness', 'mean_1stdiff', 'mean_2nddiff'],
    'bvp': ['mean', 'power', 'std', 'kurtosis', 'skewness', 'mean_1stdiff', 'mean_2nddiff'],
    'eda': ['mean', 'power', 'std', 'kurtosis', 'skewness', 'mean_1stdiff', 'mean_2nddiff', 'SCR_peakcount', 'mean_SCR_amplitude', 'mean_SCR_risetime', 'sum_SCR_amplitudes', 'sum_SCR_risetimes', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'],
    'hr': ['mean', 'power', 'std', 'kurtosis', 'skewness', 'mean_1stdiff', 'mean_2nddiff'],
}

metrics2compute=['Sen', 'FPR', 'TiW', 'AUC', 'resolution', 'reliability', 'skill']


sqi_channels = ['SQI EDA', 'SQI BVP'] # sqi channels to use
sampling_frequency = 128 # in Hz

# Constants
CATEGORICAL_PALETTE = ['#4179A0', '#A0415D',
                       '#44546A', '#44AA97', '#FFC000', '#0F3970', '#873C26']


CHANNELS = ['acc_x', 'acc_y', 'acc_z',
                'acc_mag', 'bvp', 'eda', 'hr', 'temp']