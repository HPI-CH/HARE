"""
test normalizing SampleTimeFine

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
data_config = SonarConfig(dataset_path='C:/Users/Leander/HPI/BP/UnicornML/dataset/ML Prototype Recordings')
settings.init(data_config)
random.seed(1678978086101)

# Load dataset
recordings = settings.DATA_CONFIG.load_dataset(limit_n_recs=10, multiprocessing=True, data_config=data_config)

for recording in recordings:
    print(recording.sensor_frame.shape)


