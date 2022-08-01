"""
template experiment with new config

K-Fold (Leave recording out) evaluation of JensModel saving accuracy, confusion matrix and subject information to an experiment folder
This uses the new DataConfig.
Expected accuracy: ~xx%
Takes around 6 minutes on DHC lab

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



# Init
data_config = SonarConfig(dataset_path='data/sonar-dataset')
settings.init(data_config)
random.seed(1678978086101)

# Load dataset
recordings = settings.DATA_CONFIG.load_dataset(limit_n_recs=5, multiprocessing=False)

# Preprocess
recordings = Preprocessor().jens_preprocess(recordings)

# # Save recordings - Kirill will help you to save some time
# path_preprocessed_folder = 'data/preprocessed_recordings'
# preprocessing_file_name = 'all_recordings'
# save_all_recordings(recordings, path_preprocessed_folder, preprocessing_file_name)
# recordings = load_all_recordings(os.path.join(path_preprocessed_folder, preprocessing_file_name))

# KFold cross validation (leave recordings out)
k = 2
accuracies: "list[float]" = []
recordings: np.ndarray = np.array(recordings)

window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "kfold_sonar_exp"
)

k_fold = KFold(n_splits=k, random_state=None)
recordings_subject_str = []
for idx, (train_index, test_index) in enumerate(k_fold.split(recordings)):
    print(f"Doing fold {idx}/{k-1} ... ============================================================")
    recordings_train, recordings_test = (
        recordings[train_index],
        recordings[test_index],
    )
    to_recordings_subject_str = lambda recordings: ''.join(list(map(lambda recording: recording.subject, recordings)))
    recordings_subject_str.append(f"recordings_subjects: train: {to_recordings_subject_str(recordings_train)}, test: {to_recordings_subject_str(recordings_test)}")

    model = JensModel(
        n_epochs=2,
        window_size=window_size,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
    )
    model.windowize_convert_fit(recordings_train)

    X_test, y_test_true = model.windowize_convert(recordings_test)
    y_test_pred = model.predict(X_test)

    # Evaluate model
    accuracies.append(accuracy(y_test_pred, y_test_true))
    # model.export(experiment_folder_path) # opt: export model to folder
    # create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = f"conf_matrix_fold{idx}") 


# save a simple test report to the experiment folder
result_md = f"""
# Results

| Fold | Accuracy | recordings_kfold_split_idx | |
|-|-|-|-|
"""
for idx, accuracy in enumerate(accuracies):
    result_md += f"|{idx}|{accuracy}|{recordings_subject_str[idx]}|| \n"
result_md += f"|Average|{np.mean(accuracies)}||| \n"

with open(os.path.join(experiment_folder_path, "k_fold_accuracy_results.md"), "w+") as f:
    f.writelines(result_md)
