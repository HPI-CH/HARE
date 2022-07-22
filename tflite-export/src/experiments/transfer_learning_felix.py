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

# define helper functions


def freezeNonDenseLayers(model: RainbowModel):
    # Set non dense layers to not trainable (freezing them)
    for layer in model.model.layers:
        layer.trainable = type(layer) == Dense


def map_predictions_to_indexes(y: np.ndarray) -> list:
    """
    maps the labels to one hot encoding
    """
    return list(map(lambda x: np.argmax(x), y))


# Init
data_config = GaitAnalysisConfig(dataset_path="../../data/fatigue_dual_task")
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

# define param grid for grid search
param_grid = {
    "window_size": [window_size],
    "n_features": [len(features)],
    "n_classes": [n_classes],
    "batch_size": [32],
    "n_epochs": [20],
    "learning_rate": [0.0001],
    "metrics": ["accuracy"],
    "stride_size": [window_size, window_size // 2],
}


def two_conv_model(**params) -> "GaitAnalysisTLModel":
    return GaitAnalysisTLModel(
        window_size=params.get("window_size", window_size),
        n_features=params.get("n_features", len(features)),
        n_outputs=params.get("n_classes", n_classes),
        n_epochs=params.get("n_epochs", 10),
        learning_rate=params.get("learning_rate", 0.001),
        batch_size=params.get("batch_size", 32),
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )


def conv_model(**params) -> "GaitAnalysisOGModel":

    return GaitAnalysisOGModel(
        window_size=params.get("window_size", window_size),
        n_features=params.get("n_features", len(features)),
        n_outputs=params.get("n_classes", n_classes),
        n_epochs=params.get("n_epochs", 10),
        learning_rate=params.get("learning_rate", 0.001),
        batch_size=params.get("batch_size", 32),
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )


def resnet_model(**params) -> "ResNetModel":
    return ResNetModel(
        window_size=window_size,
        n_features=len(features),
        n_outputs=n_classes,
        n_epochs=params.get("n_epochs", 10),
        learning_rate=params.get("learning_rate", 0.001),
        batch_size=params.get("batch_size", 32),
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )


def cnn_lstm(**params) -> "FranzDeepConvLSTM":
    return FranzDeepConvLSTM(
        window_size=window_size,
        n_features=len(features),
        n_outputs=n_classes,
        n_epochs=params.get("n_epochs", 10),
        learning_rate=params.get("learning_rate", 0.001),
        batch_size=params.get("batch_size", 32),
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )


# define models to test
models = [
    conv_model,
    two_conv_model,
    resnet_model,
    cnn_lstm,
]

# setup list for final boxplot
model_performances = [[] for elem in range(4 * len(models))]
print(f"Initialized boxplot list: {model_performances}...")


# iterate over subjects
for tl_sub in subs:
    result_md = f"# Experiment on {tl_sub}\n\n"
    train_subs = [sub for sub in subs if sub != tl_sub]

    # Load data
    recordings = data_config.load_dataset(subs=subs, features=features)
    recordings_train, recordings_tl = recordings.split_leave_subject_out(
        tl_sub)
    experiment_folder_path_tl = os.path.join(experiment_folder_path, tl_sub)
    os.makedirs(experiment_folder_path_tl, exist_ok=True)
    # Test Train Split
    recordings_tl_train, recordings_tl_test = recordings_tl.split_by_percentage(
        0.2)

    # downsample data
    if target_sampling_rate != None:
        print(f"...Resampling")
        recordings_train.resample(
            target_sampling_rate, base_sampling_rate, show_plot=True, path=experiment_folder_path_tl)
        recordings_tl_train.resample(target_sampling_rate, base_sampling_rate)
        recordings_tl_test.resample(target_sampling_rate, base_sampling_rate)
        print(f"...Resampling input data done")

    # Windowize
    windows_train = recordings_train.windowize(window_size)
    windows_tl_train = recordings_tl_train.windowize(window_size)
    windows_tl_test = recordings_tl_test.windowize(window_size)

    # Convert
    X_train, y_train = DataSet.convert_windows_sonar(
        windows_train, data_config.n_activities()
    )
    X_tl_train, y_tl_train = DataSet.convert_windows_sonar(
        windows_tl_train, data_config.n_activities()
    )

    X_tl_test, y_tl_test = DataSet.convert_windows_sonar(
        windows_tl_test, data_config.n_activities()
    )
    # iterate over models
    for model_idx, model in enumerate(models):
        # build model
        base_model = model()
        result_md += f"## Model: {base_model.model_name}\n\n"
        # ensure the base model implements the sklearn estimator interface
        experiment_folder_path_tl_model = os.path.join(
            experiment_folder_path_tl, base_model.model_name)
        os.makedirs(experiment_folder_path_tl_model, exist_ok=True)
        print("Mean", data_config.mean)
        print("Variance", data_config.variance)

        result_md += f"###Evaluating base model\n\n"

        base_model.fit(X_train, y_train)
        result_md += "Classification report:\n\n"

        y_true, y_test_pred_base_model = y_tl_test, base_model.predict(
            X_tl_test)
        a = tf.keras.metrics.Accuracy()
        a.update_state(
            map_predictions_to_indexes(y_true),
            map_predictions_to_indexes(y_test_pred_base_model),
        )
        result_md += f"Score on {a}: {a.result().numpy()}\n"
        f = f1_m(y_true, y_test_pred_base_model)
        result_md += f"Score on F1: {f.numpy()}\n"

        create_conf_matrix(
            experiment_folder_path_tl_model,
            y_test_pred_base_model,
            y_tl_test,
            file_name="base_model_conf_matrix",
        )
        model_performances[0 + 2 * model_idx].append(a.result().numpy())

        #model_performances[1 + 4 * model_idx].append(f.numpy())

        # create tl_model
        tl_model = base_model

        tl_model.model_name += "_tl" + tl_sub

        # freeze inner layers of tl model
        freezeNonDenseLayers(tl_model)

        result_md += f"###Evaluating tl model\n\n"

        # retrain outer layers of tl model
        tl_model.fit(X_tl_train, y_tl_train)

        result_md += "Classification report:\n\n"
        y_true, y_test_pred_tl_model = y_tl_test, tl_model.predict(X_tl_test)

        m_tl = tf.keras.metrics.Accuracy()
        m_tl.update_state(
            map_predictions_to_indexes(y_true),
            map_predictions_to_indexes(y_test_pred_tl_model),
        )
        result_md += f"score on {m_tl.name}: {m_tl.result().numpy()}\n"
        f = f1_m(y_true, y_test_pred_tl_model)
        result_md += f"Score on F1: {f.numpy()}\n"

        model_performances[1 + 2 * model_idx].append(m_tl.result().numpy())

        #model_performances[3 + 4 * model_idx].append(f.numpy())

        # create confusion matrix and text metrics for tl model
        create_conf_matrix(
            experiment_folder_path_tl_model,
            y_test_pred_tl_model,
            y_tl_test,
            file_name="tl_model_conf_matrix",
        )

        # export base model and tl model

    # saving results in markdown
    with open(os.path.join(experiment_folder_path_tl, "results.md"), "w+") as f:
        f.writelines(result_md)

# add results to boxplot
print(f"Plotting model performances: {model_performances}")
fig, ax = plt.subplots()
ax.set_title('Multiple Samples with Different sizes')
ax.boxplot(model_performances)

ax.set_title(f"Comparison of models with and without transfer learning on gait dataset. \n"
             f"The models used are: {models}")
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(['Base Acc', 'Tl Acc', 'Base Acc(1)', 'TL F1(1)',
                    'Base Acc', 'Tl Acc', 'Base Acc(1)', 'TL F1(1)',
                    'Base Acc', 'Tl Acc', 'Base Acc(1)', 'TL F1(1)',
                    'Base Acc', 'Tl Acc', 'Base Acc(1)', 'TL F1(1)'])
plt.savefig(os.path.join(experiment_folder_path, 'model_performances.png'))
plt.show()
