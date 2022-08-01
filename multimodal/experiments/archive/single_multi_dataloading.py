"""
test multi vs single thread data loading

"""

import os
import random

import numpy as np
from sklearn.model_selection import KFold
import time
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

start_single = time.time()
recordings = settings.DATA_CONFIG.load_dataset(limit_n_recs=50, multiprocessing=False)
end_single = time.time()


start_multi = time.time()
recordings = settings.DATA_CONFIG.load_dataset(limit_n_recs=50, multiprocessing=True, data_config=data_config)
end_multi = time.time()

print("single thread processing: " + str(end_single - start_single))
print("multithread processing: " + str(end_multi - start_multi))



