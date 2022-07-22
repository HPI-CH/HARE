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

# define parameters
settings.init(data_config)
window_size = (1000 * target_sampling_rate) // base_sampling_rate
n_classes = data_config.n_activities()
n_classes = data_config.n_activities()

#define path
experiment_folder_path = new_saved_experiment_folder("choose_best_model_exp_felix")

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



def init_gait_analysis_tl_model(**params) -> "GaitAnalysisTLModel":
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


def init_gait_analysis_og_model(**params) -> "GaitAnalysisOGModel":

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


def init_resnet_model(**params) -> "ResNetModel":
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


def init_franz_model(**params) -> "FranzDeepConvLSTM":
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
    init_gait_analysis_og_model,
    init_gait_analysis_tl_model,
    init_resnet_model,
    init_franz_model,
]

# setup list for final boxplot
model_training_performances = [[] for _ in range(2* len(models))]
print(f"Initialized boxplot list: {model_training_performances}...")

# setup list for final boxplot
model_lopo_performances = [[] for _ in range(2* len(models))]
print(f"Initialized boxplot list: {model_lopo_performances}...")

# iterate over subjects
for lopo_sub in subs:
    result_md = f"# Experiment on {lopo_sub}\n\n"
    train_subs = [sub for sub in subs if sub != lopo_sub]

    # Load data
    recordings = data_config.load_dataset(subs=subs, features=features)
    recordings_train, recordings_lopo = recordings.split_leave_subject_out(lopo_sub)
    experiment_folder_path_lopo = os.path.join(experiment_folder_path, lopo_sub)
    os.makedirs(experiment_folder_path_lopo, exist_ok=True)
    # downsample data
    if target_sampling_rate != None:
        print(f"...Resampling")
        recordings_train.resample(target_sampling_rate, base_sampling_rate, show_plot=True, path=experiment_folder_path_lopo)
        recordings_lopo.resample(target_sampling_rate, base_sampling_rate)
        print(f"...Resampling input data done")

    # Test Train Split
    k = 5
    split_indexes = recordings_train.intra_kfold_split(k)
    for elem in range(k):
        print(f"Running {elem}-fold cross validation...")
        result_md += f"# {elem}-fold cross validation\n\n"
        # split data into train and test
        recordings_train_fold, recordings_test_fold = recordings_train.intra_split_by_indexes(
            split_indexes[elem]
        )
        # Windowize
        print(f"...Windowizing")
        windows_train_fold = recordings_train_fold.windowize(window_size)
        windows_test_fold = recordings_test_fold.windowize(window_size)
        windows_lopo = recordings_lopo.windowize(window_size)
        print(f"...Windowizing done")
        # Convert
        X_train, y_train = DataSet.convert_windows_sonar(
            windows_train_fold, data_config.n_activities()
        )
        X_test, y_test = DataSet.convert_windows_sonar(
            windows_test_fold, data_config.n_activities()
        )

        X_lopo, y_lopo = DataSet.convert_windows_sonar(
            windows_lopo, data_config.n_activities()
        )
        # iterate over models
        for model_idx, model in enumerate(models):
            # build model
            model = model()
            result_md += f"## Model: {model.model_name}\n\n"
            # ensure the base model implements the sklearn estimator interface
            experiment_folder_path_lopo_model = os.path.join(experiment_folder_path_lopo, model.model_name)
            os.makedirs(experiment_folder_path_lopo_model, exist_ok=True)
            print("Mean", data_config.mean)
            print("Variance", data_config.variance)

            result_md += f"###Evaluating base model for fold\n\n"
            # evaluate base model
            result_md += f"##### Training\n\n"
            model.fit(X_train, y_train)
            score = model.evaluate(X_test, y_test)
            f1 = f1_m(y_test, model.predict(X_test))
            result_md += f"{model.model_name} loss: {score[0]}, accuracy: {score[1]}\n\n"
            model_training_performances[2 * model_idx].append(score[1])
            model_training_performances[2 * model_idx + 1].append(f1)

            result_md += f"##### Testing\n\n"
            result_md += "Classification report:\n\n"

            y_true, y_lopo_pred_model = y_lopo, model.predict(X_lopo)
            m = tf.keras.metrics.Accuracy()
            m.update_state(
                map_predictions_to_indexes(y_true),
                map_predictions_to_indexes(y_lopo_pred_model),
            )
            result_md += f"Score on {m}: {m.result().numpy()}\n"

            f = f1_m(y_true, y_lopo_pred_model)
            result_md += f"Score on F1: {f}\n"
            model_lopo_performances[2* model_idx].append(m.result().numpy())
            model_lopo_performances[2* model_idx + 1].append(f)

            create_conf_matrix(
                experiment_folder_path_lopo_model,
                y_lopo_pred_model,
                y_true,
                file_name="model_conf_matrix",
            )
        
        # saving results in markdown
        with open(os.path.join(experiment_folder_path_lopo, "results.md"), "w+") as f:
            f.writelines(result_md)
    
# add results to boxplot
print(f"Plotting model performances during training: {model_training_performances}")
fig, ax = plt.subplots()
ax.boxplot(model_training_performances)

ax.set_title('Comparison of differet models on gait dataset during training')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy during training')
ax.set_xticklabels(['1ConvModel', '1ConvModel-f1', '2ConvModel', '2ConvModel-f1', 'Resnet', 'Resnet-f1', 'Conv-LSTM', 'Conv-LSTM-f1'])
plt.savefig(os.path.join(experiment_folder_path, 'model_training_performances.png'))
plt.show()

# add results to boxplot
print(f"Plotting model performances: {model_lopo_performances}")
fig, ax = plt.subplots()
ax.boxplot(model_lopo_performances)

ax.set_title('Comparison of differet models on gait dataset on lopo evaluation')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy on left out person')
ax.set_xticklabels(['1ConvModel', '1ConvModel-f1', '2ConvModel', '2ConvModel-f1', 'Resnet', 'Resnet-f1', 'Conv-LSTM', 'Conv-LSTM-f1'])
plt.savefig(os.path.join(experiment_folder_path, 'model_lopo_performances.png'))
plt.show()

