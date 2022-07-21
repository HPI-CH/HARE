import os
from multiprocessing import Pool

import pandas as pd

from sonar_types import Recording
from utils import verbose, log


def save_recordings(recordings: list[Recording], path: str) -> None:
    """
    Save each recording to a csv file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for (index, recording) in enumerate(recordings):
        verbose(f'Saving recording {index} / {len(recordings)}')

        recording.activities.index = recording.sensor_frame.index

        recording_dataframe = recording.sensor_frame.copy()
        recording_dataframe['SampleTimeFine'] = recording.time_frame
        recording_dataframe['activity'] = recording.activities

        filename = str(index) + '_' + recording.subject + '.csv'
        recording_dataframe.to_csv(os.path.join(path, filename), index=False)

    log('Saved recordings to ' + path)


def save_recordings_mp(recordings: list[Recording], path: str) -> None:
    """
    Save each recording to a csv file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    pool = Pool()

    for index, recording in enumerate(recordings):
        verbose(f'Saving recording {index} / {len(recordings)}')
        pool.apply_async(save_recording, (index, recording, path))

    pool.close()
    pool.join()

    log('Saved recordings to ' + path)


def save_recording(index: int, recording: Recording, path: str):
    recording.activities.index = recording.sensor_frame.index

    recording_dataframe = recording.sensor_frame.copy()
    recording_dataframe['SampleTimeFine'] = recording.time_frame
    recording_dataframe['activity'] = recording.activities

    filename = str(index) + '_' + recording.subject + '.csv'
    recording_dataframe.to_csv(os.path.join(path, filename), index=False)


def list_cached_recordings(path: str) -> list[str]:
    recording_files = os.listdir(path)
    recording_files = list(filter(lambda file: file.endswith('.csv'), recording_files))
    return recording_files


def load_cached_recordings(path: str, limit: int = None) -> list[Recording]:
    """
    Load the recordings from a folder containing csv files.
    """
    recordings = []
    recording_files = list_cached_recordings(path)

    if limit is not None:
        recording_files = recording_files[:limit]

    recording_files = sorted(recording_files, key=lambda file: int(file.split('_')[0]))
    recording_files = list(map(lambda file_name: os.path.join(path, file_name), recording_files))

    pool = Pool()
    recordings = pool.imap_unordered(
        read_recording_from_csv, recording_files, 10
    )
    pool.close()
    pool.join()
    recordings = list(recordings)

    log(f'Loaded {len(recordings)} recordings from {path}')

    return recordings


def read_recording_from_csv(file: str) -> Recording:
    verbose(f'Loading recording {file}', end='\r')

    recording_dataframe = pd.read_csv(file)
    time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
    activities = recording_dataframe.loc[:, 'activity']
    sensor_frame = recording_dataframe.loc[:, recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
    subject = file.rsplit("/", maxsplit=1)[1].split('_')[1]
    return Recording(sensor_frame, time_frame, activities, subject)
