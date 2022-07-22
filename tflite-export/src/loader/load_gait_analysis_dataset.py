import pandas as pd
import os

from sklearn.utils import resample
from utils.Recording import Recording

def load_gait_analysis_dataset(path: str, subs: list) -> "list[Recording]":
    """
    Loads the gait analysis dataset.
    :param path: Path to the dataset.
    :param subs: List of subjects to load.
    :return: List of recordings.
    """
    recordings = []

    recording_folders = os.listdir(path)
    recording_folders = list(filter(lambda folder: "st" in folder, recording_folders))
    control_folders = list(filter(lambda folders: folders.endswith('_control'), recording_folders))
    fatigue_folders = list(filter(lambda folders: folders.endswith('_fatigue'), recording_folders))

    for (index, sub) in enumerate(subs):
        for folder in control_folders:
            print(f'Loading control recording for {folder}_{sub}')
            sub_folder = os.listdir(os.path.join(path, folder))
            sub_folder = list(filter(lambda folders: folders.startswith(sub), sub_folder))[0]
            
            LF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "LF.csv"))
            RF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "RF.csv"))
            SA_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "SA.csv"))

            GYR_LF_recording = LF_recording.loc[:,'GyrX':'GyrZ']
            GYR_LF_recording.columns = ['GYR_X_LF', 'GYR_Y_LF', 'GYR_Z_LF']
            GYR_RF_recording = RF_recording.loc[:,'GyrX':'GyrZ']
            GYR_RF_recording.columns = ['GYR_X_RF', 'GYR_Y_RF', 'GYR_Z_RF']
            GYR_SA_recording = SA_recording.loc[:,'GyrX':'GyrZ']
            GYR_SA_recording.columns = ['GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']
            ACC_LF_recording = LF_recording.loc[:,'AccX':'AccZ']
            ACC_LF_recording.columns = ['ACC_X_LF', 'ACC_Y_LF', 'ACC_Z_LF']
            ACC_RF_recording = RF_recording.loc[:,'AccX':'AccZ']
            ACC_RF_recording.columns = ['ACC_X_RF', 'ACC_Y_RF', 'ACC_Z_RF']
            ACC_SA_recording = SA_recording.loc[:,'AccX':'AccZ']
            ACC_SA_recording.columns = ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA']

            sensor_frame = pd.concat([GYR_LF_recording, ACC_LF_recording, GYR_RF_recording, ACC_RF_recording, GYR_SA_recording, ACC_SA_recording], axis=1)
            time_frame = LF_recording.loc[:, 'timestamp']
            activities = pd.Series([1] * len(GYR_LF_recording), copy=False) # non-fatigue
            subject = sub
            recordings.append(Recording(sensor_frame, time_frame, activities, subject, 2 * index - 1))
        
        for folder in fatigue_folders:
            print(f'Loading fatigue recording for {folder}_{sub}')
            sub_folder = os.listdir(os.path.join(path, folder))
            sub_folder = list(filter(lambda folders: folders.startswith(sub), sub_folder))[0]
            
            LF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "LF.csv"))
            RF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "RF.csv"))
            SA_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "SA.csv"))

            GYR_LF_recording = LF_recording.loc[:,'GyrX':'GyrZ']
            GYR_LF_recording.columns = ['GYR_X_LF', 'GYR_Y_LF', 'GYR_Z_LF']
            GYR_RF_recording = RF_recording.loc[:,'GyrX':'GyrZ']
            GYR_RF_recording.columns = ['GYR_X_RF', 'GYR_Y_RF', 'GYR_Z_RF']
            GYR_SA_recording = SA_recording.loc[:,'GyrX':'GyrZ']
            GYR_SA_recording.columns = ['GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']
            ACC_LF_recording = LF_recording.loc[:,'AccX':'AccZ']
            ACC_LF_recording.columns = ['ACC_X_LF', 'ACC_Y_LF', 'ACC_Z_LF']
            ACC_RF_recording = RF_recording.loc[:,'AccX':'AccZ']
            ACC_RF_recording.columns = ['ACC_X_RF', 'ACC_Y_RF', 'ACC_Z_RF']
            ACC_SA_recording = SA_recording.loc[:,'AccX':'AccZ']
            ACC_SA_recording.columns = ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA']

            sensor_frame = pd.concat([GYR_LF_recording, ACC_LF_recording, GYR_RF_recording, ACC_RF_recording, GYR_SA_recording, ACC_SA_recording], axis=1)
            time_frame = LF_recording.loc[:, 'timestamp']
            activities = pd.Series([0] * len(GYR_LF_recording), copy=False) # fatigue
            subject = sub
            recordings.append(Recording(sensor_frame, time_frame, activities, subject, 2 * index))
    
    print(f'Loaded {len(recordings)} recordings from {path}')
    return recordings
