from fileinput import filename
from utils.Recording import Recording
import os
import pandas as pd


def save_all_recordings(recordings: 'list[Recording]', folder_path: str, file_name: str) -> None:
    """
    Save all recordings to a single csv file.

    Refactoring idea:
    - check if there is a file already and delete it?
    """

    path = os.path.join(folder_path, f"{file_name}.csv")

    complete_dataframe = pd.DataFrame()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Initializing a main dataframe")
    for (index, recording) in enumerate(recordings):
        print(f'Adding recording {index} to the main dataframe')
        recording.activities.index = recording.sensor_frame.index
        
        recording_dataframe = recording.sensor_frame.copy()
        recording_dataframe['SampleTimeFine'] = recording.time_frame
        recording_dataframe['activity'] = recording.activities
        recording_dataframe['subject'] = recording.subject
        recording_dataframe['rec_index'] = index

        complete_dataframe = complete_dataframe.append(recording_dataframe)
    
    print(f"Saving the main dataframe to {path}, will take some time ...")
    complete_dataframe.to_csv(path, index=False)
    print('Saved recordings to ' + path)


def load_all_recordings(path_to_load: str) -> 'list[Recording]':
    """
    Load all recordings from a single csv file.
    """
    path_to_load = f"{path_to_load}.csv"

    if not os.path.exists(path_to_load):
        raise Exception(f"The dataset_path {path_to_load} does not exist.")

    print(f"Loading all recordings from {path_to_load}, will take some time ...")
    complete_dataframe = pd.read_csv(path_to_load, engine='python')
    recordings = []

    unique_rec_index = complete_dataframe['rec_index'].unique()

    for rec_index in unique_rec_index:
        print('Loading recording ' + str(rec_index))

        recording_dataframe = complete_dataframe[complete_dataframe['rec_index'] == rec_index]
        time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
        activities = recording_dataframe.loc[:, 'activity']
        subject = recording_dataframe.loc[:, 'subject'].iloc[0]
        # sensor_frame contains all columns that are not 'SampleTimeFine', 'subject', 'activity', 'rec_index'
        sensor_frame = recording_dataframe.loc[:, recording_dataframe.columns.difference(
            ['SampleTimeFine', 'subject', 'activity', 'rec_index'])]

        recordings.append(Recording( time_frame, activities, subject,sensor_frame))

    print(f'Loaded {len(recordings)} recordings from {path_to_load}')
    return recordings
