from fileinput import filename
from utils.Recording import Recording
import os
import pandas as pd
import re


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

def merge_all_gait_recordings(folder_path: str) -> None:
    """
    Merge all gait recordings from a folder into a single csv file.
    """
    for (run_index, run) in enumerate(os.listdir(folder_path)):
        run_folder_path = os.path.join(folder_path, run)
        if not re.match(r'(^OG_[dt_|st_][control|fatigue])', run):
                continue
        for (sub_index, sub) in enumerate(os.listdir(run_folder_path)):
            if not re.match(r'(^sub_[0-1][0-9])', sub):
                continue
            idx = 0
            sub_folder_path = os.path.join(run_folder_path, sub, 'cut_by_stride')
            print(f"Overwriting {os.path.join(sub_folder_path, 'merged.csv')}")
            try:
                os.remove(os.path.join(sub_folder_path, 'merged.csv'))
            except FileNotFoundError:
                pass
            complete_dataframe = pd.DataFrame()
            complete_dataframe[f"SampleTimeFine"] = pd.Series()
            print(f"Initializing a main dataframe")
            for (sensor_index, sensor) in enumerate(["LF", "RF", "SA"]):
                print(f'Adding sensor {sensor_index} to the main dataframe')
                sensor_path = os.path.join(sub_folder_path, f"{sensor}.csv")
                sensor_data = pd.read_csv(sensor_path, engine='python')
                complete_dataframe[f"GYR_X_{sensor}"] = sensor_data["GyrX"]
                complete_dataframe[f"GYR_Y_{sensor}"] = sensor_data["GyrY"]
                complete_dataframe[f"GYR_Z_{sensor}"] = sensor_data["GyrZ"]
                complete_dataframe[f"ACC_X_{sensor}"] = sensor_data["AccX"]
                complete_dataframe[f"ACC_Y_{sensor}"] = sensor_data["AccY"]
                complete_dataframe[f"ACC_Z_{sensor}"] = sensor_data["AccZ"]
                complete_dataframe[f"SampleTimeFine"] = sensor_data["timestamp"]
            activity = 0 if(run.endswith('_control')) else 1
            complete_dataframe['subject'] = sub
            complete_dataframe['activities'] = pd.Series([activity] * len(complete_dataframe[f"SampleTimeFine"]))
            complete_dataframe['rec_index'] = run_index * len(os.listdir(run_folder_path)) + sub_index
            sub_target_path = os.path.join(sub_folder_path, f"merged.csv")
            print(f"Saving the main dataframe to {sub_target_path}, will take some time ...")
            complete_dataframe.to_csv(sub_target_path, index=False)
            print('Saved recordings to ' + sub_target_path)

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

        recordings.append(Recording(sensor_frame, time_frame, activities, subject))

    print(f'Loaded {len(recordings)} recordings from {path_to_load}')
    return recordings

p = '..'
merge_all_gait_recordings(os.path.join(p, p, "data", "fatigue_dual_task"))