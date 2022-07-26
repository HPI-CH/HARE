import matplotlib.pyplot as plt
from models.RainbowModel import RainbowModel
import numpy as np
from tensorflow.keras.layers import Dense
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from models.GaitAnalysisTLModel import GaitAnalysisTLModel
from data_configs.gait_analysis_config import GaitAnalysisConfig
import utils.settings as settings
import os

# define helper functions


def map_predictions_to_indexes(y: np.ndarray) -> list:
    """
    maps the one hot encoded predictions to label indexes
    """
    return list(map(lambda x: np.argmax(x), y))


# Init
data_config = GaitAnalysisConfig(dataset_path="./data/gait")
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

# base sampling and target sampling rate
base_sampling_rate = 128
target_sampling_rate = 60

settings.init(data_config)
window_size = (1000 * target_sampling_rate) // base_sampling_rate
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder(
    "export_GaitAnalysisTL_exp")
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
# setup folder
experiment_folder_path = new_saved_experiment_folder(
    "export_GaitAnalysisTL_exp"
)

# iterate over subjects
for tl_sub in subs:
    train_subs = [sub for sub in subs if sub != tl_sub]

    # load data
    print("... loading data")
    recordings = data_config.load_dataset(subs=subs, features=features)
    print("... data loaded")

    # split data
    print("... splitting data")
    train_recordings, tl_recordings = recordings.split_leave_subject_out(
        tl_sub)
    print("... data split")

    # setup folder
    experiment_folder_path_tl = os.path.join(experiment_folder_path, tl_sub)
    os.makedirs(experiment_folder_path_tl, exist_ok=True)

    # downsample data
    if target_sampling_rate != None:
        print(f"...Resampling")
        train_recordings.resample(target_sampling_rate, base_sampling_rate)
        print(f"...Resampling input data done")

    # windowize
    print(f"...Windowizing")
    train_recordings = train_recordings.windowize(window_size)
    print(f"...Windowizing done")

    # convert windows
    print(f"...Converting windows")
    X, y = DataSet.convert_windows_sonar(
        train_recordings, data_config.n_activities())
    print(f"...Converting windows done")

    # setup model
    model = GaitAnalysisTLModel(
        window_size=window_size,
        n_features=len(features),
        n_outputs=n_classes,
        n_epochs=10,
        learning_rate=0.001,
        batch_size=32,
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description=f"GaitAnalysis transfer learning model on sub {tl_sub} for gait dataset",
    )

    # train model
    print(f"...Training model")
    model.fit(X, y)
    print(f"...Training model done")

    # freeze non dense layers
    print(f"...Freezing non dense layers")

    model.freeze_non_dense_layers()
    print(f"...Freezing non dense layers done")

    # export model
    print(f"...Exporting model")
    model.model_name = f"CNNTL_model_{tl_sub}"
    model.export(
        experiment_folder_path_tl,
        features=features,
        device_tags=data_config.sensor_suffix_order,
        class_labels=list(data_config.category_labels.keys()),
    )
    print(f"...Exporting model done")
