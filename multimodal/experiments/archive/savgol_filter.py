
import os
import random

from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_dataset import load_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from scipy.signal import savgol_filter
import pandas as pd
import copy


settings.init("opportunity")

# Load data
recordings = load_opportunity_dataset(settings.opportunity_dataset_path) 
random.seed(1678978086101)
random.shuffle(recordings)


# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# params to try
window_sizes = [5,7,11,35] 
polynoms = [2,3,5,8,12,17] 

recordings_to_try = [(recordings,0,0)]

# create recording array for each filtering param tuples
for window_size in window_sizes:
    for polynom in polynoms:
        if polynom > window_size/2:  
            continue
        recording_filtered = copy.deepcopy(recordings)
        for recording in recording_filtered:
            clmns = recording.sensor_frame.columns.values 
            recording.sensor_frame = pd.DataFrame(savgol_filter(recording.sensor_frame, window_size, polynom),columns=clmns)
        recordings_to_try.append((recording_filtered, window_size, polynom))


# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('savgol') # create folder to store results

# TODO: save/ load preprocessed data
for recordings_tuple in recordings_to_try:
    recordings = recordings_tuple[0]
    # Test Train Split
    test_percentage = 0.4
    recordings_train, recordings_test = split_list_by_percentage(recordings, test_percentage)

    # Init, Train
    model = JensModel(window_size=25, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6, verbose=1, n_epochs=10)
    model.windowize_convert_fit(recordings_train)

    # Test, Evaluate
    # labels are always in vector format
    X_test, y_test_true = model.windowize_convert(recordings_test)
    y_test_pred = model.predict(X_test)

    with open(os.path.join(experiment_folder_path, 'metrics.txt'), "a") as f:
        f.write("window size: "+str(recordings_tuple[1])+ ", polynom: " + str(recordings_tuple[2]) + "\n")
        f.write(f"{accuracy.__name__}: {accuracy(y_test_pred, y_test_true)}\n")
        f.write("\n")