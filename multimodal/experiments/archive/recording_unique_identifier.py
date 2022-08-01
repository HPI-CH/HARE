import os
import random
from loader.load_sonar_dataset import load_sonar_dataset as load_dataset
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score


settings.init("opportunity")

# Load data from opportunity to test 
recordings = load_opportunity_dataset("C:/Users/Leander/HPI/BP/UnicornML/dataset/OpportunityUCIDataset") 

# Load data from our dataset
recordings = load_dataset("C:/Users/Leander/HPI/BP/UnicornML/dataset/ML Prototype Recordings", limit=None) 

random.seed(1678978086101)
random.shuffle(recordings)

for recording in recordings:
    print(recording.recording_index)
