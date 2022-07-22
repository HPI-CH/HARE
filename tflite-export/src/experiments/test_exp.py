"""
Windowizer, Converter, new structure, working version
"""
import matplotlib.pyplot as plt
from itertools import product
from tkinter import N
from models.RainbowModel import RainbowModel
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from models.GaitAnalysisTLModel import GaitAnalysisTLModel
from data_configs.sonar_lab_config import SonarLabConfig
from data_configs.gait_analysis_config import GaitAnalysisConfig
from models.GaitAnalysisOGModel import GaitAnalysisOGModel
from models.ResNetModel import ResNetModel
from models.FranzDeepConvLSTM import FranzDeepConvLSTM
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense
from utils.metrics import f1_m
from utils.grid_search_cv import GridSearchCV
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import os



# Init
data_config = GaitAnalysisConfig(dataset_path="../../data/fatigue_dual_task")

# base sampling and target sampling rate
base_sampling_rate = 128
target_sampling_rate = 60

# define parameters
settings.init(data_config)
window_size = (1000 * target_sampling_rate) // base_sampling_rate
n_classes = data_config.n_activities()
n_classes = data_config.n_activities()

features = [
    "GYR_X_LF",
    "GYR_Y_LF",
    "GYR_Z_LF",
    "ACC_X_LF",
    "ACC_Y_LF",
    "ACC_Z_LF",
    "GYR_X_RF",
    "GYR_Y_RF",
    "GYR_Z_RF",
    "ACC_X_RF",
    "ACC_Y_RF",
    "ACC_Z_RF",
    "GYR_X_SA",
    "GYR_Y_SA",
    "GYR_Z_SA",
    "ACC_X_SA",
    "ACC_Y_SA",
    "ACC_Z_SA",
]
# define subjects
subs = [
    "sub_01",
    "sub_02",
    "sub_03",
    "sub_05",
    "sub_06",
    "sub_07",
    "sub_08",
    "sub_09",
    "sub_10",
    "sub_11",
    "sub_12",
    "sub_13",
    "sub_14",
    "sub_15",
    "sub_17",
    "sub_18",
]

# load data

#recordings = data_config.load_dataset(features=features)
# print("Number of recordings in Sonar-Lab Data-Set: ", len(recordings))
# windows = recordings.windowize(window_size=window_size)
# print("Number of windows in Sonar-Lab Data-Set: ", len(windows))


# define model
model = GaitAnalysisTLModel(
    window_size=window_size,
    n_outputs=n_classes,
    n_features=len(features),
    n_epochs=10,
    learning_rate=0.0001,
    batch_size=32,
    input_distribution_mean=data_config.mean,
    input_distribution_variance=data_config.variance,
    author="TobiUndFelix",
    version="0.1",
    description="ResNet Model for Sonar22 Dataset",
)
experiment_folder_path=new_saved_experiment_folder(
    experiment_name="gait_analysis_cnn")
model.export(
    experiment_folder_path,
    features=features,
    device_tags=data_config.sensor_suffix_order,
    class_labels=list(data_config.category_labels.keys()),
)