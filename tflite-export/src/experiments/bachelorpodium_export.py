"""
Windowizer, Converter, new structure, working version
"""
from curses import window
import random
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from models.FranzDeepConvLSTM import FranzDeepConvLSTM
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from models.ResNetModel import ResNetModel
from data_configs.sonar_lab5_config import SonarLab5Config
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from sklearn.utils import shuffle
import os

# Init
# OpportunityConfig(dataset_path='../../data/opportunity-dataset')
# Sonar22CategoriesConfig(dataset_path="../../data/filtered_dataset_without_null")
data_config = SonarLab5Config(
    dataset_path="../../data/lab_data_filtered_without_null"
)  # ["Quat_W_LF","Quat_X_LF","Quat_Y_LF","Quat_Z_LF","dq_W_LF","dq_X_LF","dq_Y_LF","dq_Z_LF","dv[1]_LF","dv[2]_LF","dv[3]_LF","Mag_X_LF","Mag_Y_LF","Mag_Z_LF","Quat_W_LW","Quat_X_LW","Quat_Y_LW","Quat_Z_LW","dq_W_LW","dq_X_LW","dq_Y_LW","dq_Z_LW","dv[1]_LW","dv[2]_LW","dv[3]_LW","Mag_X_LW","Mag_Y_LW","Mag_Z_LW","Quat_W_ST","Quat_X_ST","Quat_Y_ST","Quat_Z_ST","dq_W_ST","dq_X_ST","dq_Y_ST","dq_Z_ST","dv[1]_ST","dv[2]_ST","dv[3]_ST","Mag_X_ST","Mag_Y_ST","Mag_Z_ST","Quat_W_RW","Quat_X_RW","Quat_Y_RW","Quat_Z_RW","dq_W_RW","dq_X_RW","dq_Y_RW","dq_Z_RW","dv[1]_RW","dv[2]_RW","dv[3]_RW","Mag_X_RW","Mag_Y_RW","Mag_Z_RW","Quat_W_RF","Quat_X_RF","Quat_Y_RF","Quat_Z_RF","dq_W_RF","dq_X_RF","dq_Y_RF","dq_Z_RF","dv[1]_RF","dv[2]_RF","dv[3]_RF","Mag_X_RF","Mag_Y_RF","Mag_Z_RF"]
features = ["dq_W_LF", "dq_X_LF", "dq_Y_LF", "dq_Z_LF", "dv[1]_LF", "dv[2]_LF", "dv[3]_LF", "Mag_X_LF", "Mag_Y_LF", "Mag_Z_LF", "dq_W_LW", "dq_X_LW", "dq_Y_LW", "dq_Z_LW", "dv[1]_LW", "dv[2]_LW", "dv[3]_LW", "Mag_X_LW", "Mag_Y_LW", "Mag_Z_LW", "dq_W_ST", "dq_X_ST", "dq_Y_ST", "dq_Z_ST", "dv[1]_ST",
            "dv[2]_ST", "dv[3]_ST", "Mag_X_ST", "Mag_Y_ST", "Mag_Z_ST", "dq_W_RW", "dq_X_RW", "dq_Y_RW", "dq_Z_RW", "dv[1]_RW", "dv[2]_RW", "dv[3]_RW", "Mag_X_RW", "Mag_Y_RW", "Mag_Z_RW", "dq_W_RF", "dq_X_RF", "dq_Y_RF", "dq_Z_RF", "dv[1]_RF", "dv[2]_RF", "dv[3]_RF", "Mag_X_RF", "Mag_Y_RF", "Mag_Z_RF"]

settings.init(data_config)
window_size = 900  # 30 * 3
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder(
    f"bachelorpodium_exp_6activities")

# Load data
recordings_train = data_config.load_dataset(
    features=features)


random.seed(1678978086101)
# random.shuffle(recordings)

# Test Train Split
# recordings_train, recordings_test = recordings.split_by_percentage(
#    test_percentage=0.2)
# Windowize
windows_train = recordings_train.windowize(window_size)

# Convert
X_train, y_train = DataSet.convert_windows_sonar(
    windows_train, data_config.n_activities()
)
X_train, y_train = shuffle(X_train, y_train, random_state=42)


# or JensModel
model = ResNetModel(
    window_size=window_size,
    n_features=len(features),
    n_outputs=n_classes,
    n_epochs=10,
    learning_rate=0.0001,
    batch_size=32,
    input_distribution_mean=data_config.mean,
    input_distribution_variance=data_config.variance,
    author="TobiUndFelix",
    version="0.1",
    description=f"ResNet Model for Sonar6 Dataset. Trained on sonar lab data including all data.",
    name=f"model_including_all_data"
)

print("MEan", data_config.mean)
print("Variance", data_config.variance)

model.fit(X_train, y_train)

model.freeze_non_dense_layers()

model.export(
    experiment_folder_path,
    features=features,
    device_tags=data_config.sensor_suffix_order,
    class_labels=list(data_config.category_labels.keys()),
)
print(f"Saved to {experiment_folder_path}")
