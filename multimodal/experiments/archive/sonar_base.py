"""
test with new config

"""

import os
import random

import numpy as np
from sklearn.model_selection import KFold

import utils.settings as settings
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from loader.Preprocessor import Preprocessor
from models.JensModel import JensModel
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import SonarConfig
from utils.save_all_recordings import save_all_recordings, load_all_recordings
from utils.array_operations import split_list_by_percentage

# Init
data_config = SonarConfig(dataset_path='data/sonar-dataset')
settings.init(data_config)
random.seed(1678978086101)

# Load dataset
recordings = settings.DATA_CONFIG.load_dataset(limit_n_recs=10, multiprocessing=False)

# Preprocess
recordings = Preprocessor().our_preprocess(recordings)


window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = settings.DATA_CONFIG.n_activities()

# Test, Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

model = JensModel(
    n_epochs=2,
    window_size=100,
    n_features=n_features,
    n_outputs=n_outputs,
    batch_size=64,
)
model.windowize_convert_fit(recordings_train)

X_test, y_test_true = model.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Evaluate model
exp_accuracy = accuracy(y_test_pred, y_test_true)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "kfold_sonar_exp"
)

# save a simple test report to the experiment folder
result_md = f"""
Accuracy:
    {exp_accuracy}
"""

with open(os.path.join(experiment_folder_path, "results.md"), "w+") as f:
    f.writelines(result_md)
