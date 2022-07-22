"""
Windowizer, Converter, new structure, working version
"""
import matplotlib.pyplot as plt
from itertools import product
from tkinter import N

from sklearn.model_selection import KFold
from models.RainbowModel import RainbowModel
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from data_configs.sonar_lab_config import SonarLabConfig
from models.DeepConvLSTM import DeepConvLSTM
from models.ResNetModel import ResNetModel
from models.FranzDeepConvLSTM import FranzDeepConvLSTM
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from sklearn.utils import shuffle
from utils.metrics import f1_m
from utils.grid_search_cv import GridSearchCV
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import random
import os
import keras.backend as K

# define helper functions


def map_predictions_to_indexes(y: np.ndarray) -> list:
    """
    maps the labels to one hot encoding
    """
    return list(map(lambda x: np.argmax(x), y))

# Init


data_config = SonarLabConfig(
    dataset_path="../../data/lab_data_filtered_without_null"
)
features = ["dq_W_LF", "dq_X_LF", "dq_Y_LF", "dq_Z_LF", "dv[1]_LF", "dv[2]_LF", "dv[3]_LF", "Mag_X_LF", "Mag_Y_LF", "Mag_Z_LF", "dq_W_LW", "dq_X_LW", "dq_Y_LW", "dq_Z_LW", "dv[1]_LW", "dv[2]_LW", "dv[3]_LW", "Mag_X_LW", "Mag_Y_LW", "Mag_Z_LW", "dq_W_ST", "dq_X_ST", "dq_Y_ST", "dq_Z_ST", "dv[1]_ST",
            "dv[2]_ST", "dv[3]_ST", "Mag_X_ST", "Mag_Y_ST", "Mag_Z_ST", "dq_W_RW", "dq_X_RW", "dq_Y_RW", "dq_Z_RW", "dv[1]_RW", "dv[2]_RW", "dv[3]_RW", "Mag_X_RW", "Mag_Y_RW", "Mag_Z_RW", "dq_W_RF", "dq_X_RF", "dq_Y_RF", "dq_Z_RF", "dv[1]_RF", "dv[2]_RF", "dv[3]_RF", "Mag_X_RF", "Mag_Y_RF", "Mag_Z_RF"]

settings.init(data_config)
window_size = 900  # 30 * 3
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder("transfer_learning_lab")

# Load data
recordings = data_config.load_dataset(features=features)

print(recordings.count_activities_per_subject())
recordings.plot_activities_per_subject(experiment_folder_path, "sonarLab_activitiesPerSubject.png",
                                       "Measurement time stamps per activity for each subject in SONAR lab data set")


random.seed(1678978086101)
random.shuffle(recordings)


# get subs
subs = recordings.get_people_in_recordings()

# define models


def resnet_model() -> ResNetModel:
    model = ResNetModel(window_size=window_size,
                        n_features=len(features),
                        n_outputs=n_classes,
                        n_epochs=20,
                        learning_rate=0.0001,
                        batch_size=32,
                        input_distribution_mean=data_config.mean,
                        input_distribution_variance=data_config.variance,
                        author="TobiUndFelix",
                        version="0.1",
                        description="ResNet Model for Sonar Lab Dataset",)
    return model


def franz_deep_conv_lstm() -> FranzDeepConvLSTM:
    model = FranzDeepConvLSTM(window_size=window_size,
                              n_features=len(features),
                              n_outputs=n_classes,
                              n_epochs=20,
                              learning_rate=0.0001,
                              batch_size=32,
                              input_distribution_mean=data_config.mean,
                              input_distribution_variance=data_config.variance,
                              author="TobiUndFelix",
                              version="0.1",
                              description="Franz' cnn lstm Model for Sonar Lab Dataset",)
    return model


def deep_conv_lstm() -> DeepConvLSTM:
    model = DeepConvLSTM(window_size=window_size,
                         n_features=len(features),
                         n_outputs=n_classes,
                         n_epochs=20,
                         learning_rate=0.0001,
                         batch_size=32,
                         input_distribution_mean=data_config.mean,
                         input_distribution_variance=data_config.variance,
                         author="TobiUndFelix",
                         version="0.1",
                         description="DeepConvLSTM Model for Sonar Lab Dataset",)
    return model


models = [
    resnet_model,
    franz_deep_conv_lstm,
    deep_conv_lstm,
]


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# setup list for final boxplot
model_accuracies = [[] for _ in range(2 * len(models))]
model_f1Scores = [[] for _ in range(2 * len(models))]

print(f"Initialized boxplot list: {model_accuracies}...")

print("Starting experiment...")
for model_idx, model in enumerate(models):

    result_md = f"# Experiment on {model.__name__}\n\n"

    experiment_model_folder_path = os.path.join(
        experiment_folder_path, model.__name__)
    os.makedirs(experiment_model_folder_path, exist_ok=True)
    for tl_sub in subs:

        # setup directory
        experiment_model_tl_folder_path = os.path.join(
            experiment_model_folder_path, tl_sub)
        os.makedirs(experiment_model_tl_folder_path, exist_ok=True)

        # split data
        recordings_tl, recordings_train = recordings.split_by_subjects([
                                                                       tl_sub])
        if len(recordings_tl) < 5:
            print(f"Skipping {tl_sub} because it has less than 5 recordings")
            continue
        # setup splits
        kf = KFold(n_splits=5)
        for idx, (train_index, test_index) in enumerate(kf.split(recordings_tl)):
            result_md += f"## Split {idx + 1}\n\n"
            result_md += f"Training on {len(train_index)} recordings\n"
            # split data
            recordings_tl_train, recordings_tl_test = recordings_tl.split_by_indexes(
                train_index)

            # Windowize
            windows_train = recordings_train.windowize(window_size)
            windows_tl_train = recordings_tl_train.windowize(window_size)
            windows_tl_test = recordings_tl_test.windowize(window_size)

            #  Convert
            X_train, y_train = DataSet.convert_windows_sonar(
                windows_train, data_config.n_activities()
            )
            X_tl_train, y_tl_train = DataSet.convert_windows_sonar(
                windows_tl_train, data_config.n_activities()
            )

            X_tl_test, y_tl_test = DataSet.convert_windows_sonar(
                windows_tl_test, data_config.n_activities()
            )

            # init model
            base_model = model()
            result_md += f"## Sub to transfer learn on: {tl_sub}\n\n"
            result_md += f"### Evaluating base model\n\n"
            result_md += f"{base_model.model.summary()}\n\n"
            # train model
            base_model.fit(X_train, y_train)
            # evaluate model
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
                experiment_model_tl_folder_path,
                y_test_pred_base_model,
                y_tl_test,
                file_name=f"base_model_conf_matrix_fold_{idx}",
            )
            model_accuracies[2 * model_idx].append(a.result().numpy())
            model_f1Scores[2 * model_idx].append(f)

            # create tl_model
            tl_model = base_model

            tl_model.model_name += "_tl" + tl_sub

            # freeze inner layers of tl model
            tl_model.freezeNonDenseLayers()

            result_md += f"###Evaluating tl model\n\n"
            result_md += f"{tl_model.model.summary()}\n\n"

            # retrain outer layers of tl model
            print(f"Retraining tl model on {tl_sub}")
            tl_model.fit(X_tl_train, y_tl_train)

            result_md += "Classification report:\n\n"
            y_true, y_test_pred_tl_model = y_tl_test, tl_model.predict(
                X_tl_test)

            m_tl = tf.keras.metrics.Accuracy()
            m_tl.update_state(
                map_predictions_to_indexes(y_true),
                map_predictions_to_indexes(y_test_pred_tl_model),
            )
            result_md += f"score on {m_tl.name}: {m_tl.result().numpy()}\n"
            f = f1_m(y_true, y_test_pred_tl_model)
            result_md += f"Score on F1: {f.numpy()}\n"

            model_accuracies[1 + 2 * model_idx].append(m_tl.result().numpy())
            model_f1Scores[1 + 2 * model_idx].append(f)
            # create confusion matrix and text metrics for tl model
            create_conf_matrix(
                experiment_model_tl_folder_path,
                y_test_pred_tl_model,
                y_tl_test,
                file_name=f"tl_model_conf_matrix_fold_{idx}",
            )
    # save result_md
    with open(os.path.join(experiment_model_folder_path, f"results.md"), "w") as f:
        f.write(result_md)

# add results to boxplot
print(f"Plotting model performances: {model_accuracies}")
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.boxplot(model_accuracies)
ax.set_title(
    f"Comparison of model accuracy with and without transfer learning on sonar lab data-set\n")
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_xticklabels(['ResNet', 'ResNet-TL', 'CNNLSTM',
                   'CNNLSTM-TL', 'DeepConv-\nLSTM', 'DeepConv-\nLSTM-TL'])
plt.savefig(os.path.join(experiment_folder_path, 'model_accuracies.png'))

plt.clf()
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.boxplot(model_f1Scores)
ax.set_title(
    f"Comparison of model f1-scores with and without transfer learning on sonar lab data-set\n")
ax.set_xlabel('Models')
ax.set_ylabel('F1-Score')
ax.set_xticklabels(['ResNet', 'ResNet-TL', 'CNNLSTM',
                   'CNNLSTM-TL', 'DeepConv-\nLSTM', 'DeepConv-\nLSTM-TL'])
plt.savefig(os.path.join(experiment_folder_path, 'model_f1Scores.png'))
plt.show()
plt.close()

# calculate improvement of tl model
diff_resnet_tl = np.mean(model_accuracies[0]) - np.mean(model_accuracies[1])
diff_cnnlstm_tl = np.mean(model_accuracies[2]) - np.mean(model_accuracies[3])
diff_deepconvlstm_tl = np.mean(
    model_accuracies[4]) - np.mean(model_accuracies[5])
avg_improvement = (diff_resnet_tl + diff_cnnlstm_tl + diff_deepconvlstm_tl) / 3

# save results to file
model_performances_file_path = os.path.join(
    experiment_folder_path, "model_performances.txt")
with open(model_performances_file_path, "w") as f:
    f.write(f"\n\nModel performances:\n")
    f.write(f"ResNet: {model_accuracies[0]}\n")
    f.write(f"ResNet-TL: {model_accuracies[1]}\n")
    f.write(f"CNNLSTM: {model_accuracies[2]}\n")
    f.write(f"CNNLSTM-TL: {model_accuracies[3]}\n")
    f.write(f"DeepConvLSTM: {model_accuracies[4]}\n")
    f.write(f"DeepConvLSTM-TL: {model_accuracies[5]}\n")
    f.write(f"\n\nImprovement of tl model:\n")
    f.write(f"ResNet-TL: {diff_resnet_tl}\n")
    f.write(f"CNNLSTM-TL: {diff_cnnlstm_tl}\n")
    f.write(f"DeepConvLSTM-TL: {diff_deepconvlstm_tl}\n")
    f.write(f"\n\)")
    f.write(f"\n\nAverage improvement of tl model: {avg_improvement}")
