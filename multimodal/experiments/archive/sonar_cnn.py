import os
import random

import numpy as np
from sklearn.model_selection import KFold

import utils.settings as settings
from evaluation.metrics import accuracy
from loader.Preprocessor import Preprocessor
from loader.load_dataset import load_dataset
from models.JensModel import JensModel
from training.k_fold_test import k_fold_cross_validation_test
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder

# Init
settings.init("sonar")
random.seed(1678978086101)

print(settings.ACTIVITIES)
# Load & Prepare data

# Pass a limit to only load a few recordings (for testing)
recordings = load_dataset("/Users/franz/Projects/BP/new_data", limit=None)

activities = len(settings.LABELS)

recordings = Preprocessor().our_preprocess(recordings)

# Filter activities & remove all recordings which have no matching activity

# 8: Wagen schieben, 15: Haare kämmen, 29: Wäsche im Bett, 35: Essen austragen
# 39: Getränke ausschenken, 52: Rollstuhl schieben, 55: Computerarbeit
recordings = filter_activities(recordings, [8, 15, 29, 35, 39, 52, 55])
recordings = list(filter(lambda rec: len(rec.sensor_frame) > 60, recordings))

# Training / Evaluation with K-Fold Crossvalidation
sensors = recordings[0].sensor_frame.shape[1]
n_outputs = len(settings.ACTIVITIES)
k = 5

accuracies: "list[float]" = []
recordings: np.ndarray = np.array(recordings)

k_fold = KFold(n_splits=k, random_state=None)
for idx, (train_index, test_index) in enumerate(k_fold.split(recordings)):
    print(f"Starting fold {idx}")
    recordings_train, recordings_test = (
        recordings[train_index],
        recordings[test_index],
    )

    model = JensModel(
        epochs=5,
        window_size=100,
        n_features=sensors,
        n_outputs=n_outputs,
        batch_size=64,
    )
    model.windowize_convert_fit(recordings_train)

    X_test, y_test = model.windowize_convert(recordings_test)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracies.append(accuracy(y_pred, y_test))

# Save Evaluation

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "our_data_jens_cnn"
)  # create folder to store results


# Maybe use markdown report, but I'm not sure yet what we want to pass
result_md = f"""
# Results

| Fold | Accuracy |  | |
|-|-|-|-|
"""

for idx, accuracy in enumerate(accuracies):
    result_md += f"|{idx+1}|{accuracy}||| \n"

with open(os.path.join(experiment_folder_path, "results.md"), "w+") as f:
    f.writelines(result_md)
