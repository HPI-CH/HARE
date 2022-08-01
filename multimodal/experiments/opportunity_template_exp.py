"""
Windowizer, Converter, new structure, working version
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter

from models.JensModel import JensModel
from models.MultilaneConv import MultilaneConv
from models.BestPerformerConv import BestPerformerConv
from models.OldLSTM import OldLSTM
from models.SenselessDeepConvLSTM import SenselessDeepConvLSTM
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from utils.DataConfig import OpportunityConfig


experiment_name = "opportunity_template_exp"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Init
data_config = OpportunityConfig(dataset_path="data/opportunity-dataset")
settings.init(data_config)
window_size = 30 * 3
n_classes = 6

# Lib -----------------------------------------------------------
leave_recording_out_split = lambda test_percentage: lambda recordings: split_list_by_percentage(
    list_to_split=recordings, percentage_to_split=test_percentage
)
# leave_recording_out_split(test_percentage=0.3)(recordings)
def leave_person_out_split_idx(recordings, test_person_idx):
    subset_from_condition = lambda condition, recordings: [
        recording for recording in recordings if condition(recording)
    ]
    recordings_train = subset_from_condition(
        lambda recording: recording.subject != test_person_idx, recordings
    )
    recordings_test = subset_from_condition(
        lambda recording: recording.subject == test_person_idx, recordings
    )
    return recordings_train, recordings_test


leave_person_out_split = lambda test_person_idx: lambda recordings: leave_person_out_split_idx(
    recordings=recordings, test_person_idx=test_person_idx
)
# leave_person_out_split(test_person_idx=2)(recordings) # 1-4, TODO: could be random


# Config --------------------------------------------------------------------------------------------------------------
preprocess = lambda recordings: Preprocessor().jens_preprocess_with_normalize(
    recordings
)
windowize = lambda recordings: Windowizer(window_size=window_size).jens_windowize(
    recordings
)
convert = lambda windows: Converter(n_classes=n_classes).sonar_convert(windows)
flatten = lambda tuple_list: [item for sublist in tuple_list for item in sublist]
test_train_split = lambda recordings: leave_person_out_split(test_person_idx=2)(
    recordings
)


# Load data
recordings = settings.DATA_CONFIG.load_dataset()

random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing
recordings = preprocess(recordings)

# Test Train Split
recordings_train, recordings_test = test_train_split(recordings)

# Windowize
windows_train, windows_test = windowize(recordings_train), windowize(recordings_test)

# Convert
X_train, y_train, X_test, y_test = tuple(
    flatten(map(convert, [windows_train, windows_test]))
)

# or JensModel
model = LeanderDeepConvLSTM(
    window_size=window_size,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=n_classes,
    n_epochs=5,
    learning_rate=0.001,
    batch_size=32,
    wandb_config={
        "project": "all_experiments_project",
        "entity": "valentindoering",
        "name": experiment_name,
    },
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    experiment_name
)  # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
