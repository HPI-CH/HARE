"""
Windowizer, Converter, new structure, working version
"""
import random
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from models.ResNetModel import ResNetModel
from data_configs.sonar_lab_config import SonarLabConfig
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from sklearn.utils import shuffle
import os
# Init
data_config = SonarLabConfig(
    dataset_path="./data/nursing"
)
features = ["dq_W_LF", "dq_X_LF", "dq_Y_LF", "dq_Z_LF", "dv[1]_LF", "dv[2]_LF", "dv[3]_LF", "Mag_X_LF", "Mag_Y_LF", "Mag_Z_LF", "dq_W_LW", "dq_X_LW", "dq_Y_LW", "dq_Z_LW", "dv[1]_LW", "dv[2]_LW", "dv[3]_LW", "Mag_X_LW", "Mag_Y_LW", "Mag_Z_LW", "dq_W_ST", "dq_X_ST", "dq_Y_ST", "dq_Z_ST", "dv[1]_ST",
            "dv[2]_ST", "dv[3]_ST", "Mag_X_ST", "Mag_Y_ST", "Mag_Z_ST", "dq_W_RW", "dq_X_RW", "dq_Y_RW", "dq_Z_RW", "dv[1]_RW", "dv[2]_RW", "dv[3]_RW", "Mag_X_RW", "Mag_Y_RW", "Mag_Z_RW", "dq_W_RF", "dq_X_RF", "dq_Y_RF", "dq_Z_RF", "dv[1]_RF", "dv[2]_RF", "dv[3]_RF", "Mag_X_RF", "Mag_Y_RF", "Mag_Z_RF"]

settings.init(data_config)
window_size = 900
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder(
    f"export_resnet_exp_leave_rec_out")

# Load data
recordings = data_config.load_dataset(
    features=features)


random.seed(1678978086101)
random.shuffle(recordings)

subs = ["orhan", "valentin", "daniel"]

for sub in subs:
    print(
        f"Starting training of baseline model for subject {sub} (excluding data from {sub})")
    sub_folder_path = os.path.join(experiment_folder_path, sub)
    os.makedirs(sub_folder_path, exist_ok=True)
    # Test Train Split
    recordings_train, recordings_test = recordings.split_leave_subject_out(
        sub)
    # Windowize
    windows_train = recordings_train.windowize(window_size)
    windows_test = recordings_test.windowize(window_size)

    # Convert
    X_train, y_train = DataSet.convert_windows_sonar(
        windows_train, data_config.n_activities()
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = DataSet.convert_windows_sonar(
        windows_test, data_config.n_activities())

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
        description=f"ResNet Model for Sonar22 Dataset. Trained on sonar lab data ",
        name=f"model"
    )

    model.fit(X_train, y_train)
    print(f"Training done. Evaluating on left out subject {sub}")
    y_test_pred = model.predict(X_test)

    create_conf_matrix(sub_folder_path, y_test_pred, y_test,
                       title=f"Predictions of model trained without {sub} on data from {sub}")
    create_text_metrics(
        sub_folder_path, y_test_pred, y_test, [accuracy]
    )

    model.freeze_non_dense_layers()

    model.export(
        sub_folder_path,
        features=features,
        device_tags=data_config.sensor_suffix_order,
        class_labels=list(data_config.category_labels.keys()),
    )
    print(f"Saved to {sub_folder_path}")
