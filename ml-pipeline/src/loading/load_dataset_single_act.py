import os
import string
from itertools import repeat
from random import shuffle

import pandas as pd

from loading.XSensRecordingReaderOld import XSensRecordingReaderOld
from sonar_types import Recording
from utils.config import Config
from utils.path_helpers import get_subfolder_names


def load_dataset_old(dataset_path: str) -> list[Recording]:
    """
    Returns a list of the raw recordings SHUFFLED (independent from directory structure) (activities, subjects included, None values) (different representaion of dataset)


    This function knows the structure of the XSens dataset.
    It will call the create_recording_frame function on every recording folder.

    bp/data
        dataset_01
            activity_01
                subject_01
                    recording_01
                        random_bullshit_folder
                        sensor_01.csv
                        sensor_02.csv
                        ...
                    recording_02
                        ...
                subject_02
                    ....
            activity_02
                ...
        data_set_02
            ...

    """

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: list[Recording] = []

    # activity
    activity_folder_names = get_subfolder_names(dataset_path)
    for activity_folder_name in activity_folder_names:
        activity_folder_path = os.path.join(dataset_path, activity_folder_name)

        # subject
        subject_folder_names = get_subfolder_names(activity_folder_path)
        for subject_folder_name in subject_folder_names:
            subject_folder_path = os.path.join(activity_folder_path, subject_folder_name)

            # recording
            recording_folder_names = get_subfolder_names(subject_folder_path)
            for recording_folder_name in recording_folder_names:
                if recording_folder_name.startswith("_"):
                    continue
                recording_folder_path = os.path.join(subject_folder_path, recording_folder_name)
                # print("Reading recording: {}".format(recording_folder_path))

                recordings.append(create_recording(recording_folder_path, activity_folder_name, subject_folder_name))

    # void function, otherwise the date would have an order the same as the folder structure
    shuffle(recordings)

    return recordings


def load_dataset_synthetic(dataset_path: str) -> list[Recording]:
    """
        dataset_01
            recording_01_activity_name
                random_bullshit_folder
                sensor_01.csv
                sensor_02.csv
                ...
            ...

    """

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: list[Recording] = []

    # recording
    recording_folder_names = get_subfolder_names(dataset_path)
    for recording_folder_name in recording_folder_names:
        if recording_folder_name.startswith("_"):
            continue

        activity = recording_folder_name.split('-')[1].rstrip(string.digits)
        recording_folder_path = os.path.join(dataset_path, recording_folder_name)
        # print("Reading recording: {}".format(recording_folder_path))

        recordings.append(create_recording(recording_folder_path, activity, 'synthetic'))

    # void function, otherwise the date would have an order the same as the folder structure
    shuffle(recordings)

    return recordings


def create_recording(recording_folder_path: str, activity: str, subject: str) -> Recording:
    """
    Returns a recording
    Gets a XSens recorind folder path, loops over sensor files, concatenates them, adds activity and subject, returns a recording
    """

    raw_recording_frame = XSensRecordingReaderOld.get_recording_frame(recording_folder_path)

    time_column_name = 'SampleTimeFine'
    time_frame = raw_recording_frame[time_column_name]

    sensor_frame = raw_recording_frame.drop([time_column_name], axis=1)
    sensor_frame = reorder_sensor_columns(sensor_frame)

    activities = pd.Series(data=repeat(activity, len(time_frame)))
    subject = abs(hash(subject)) % (10 ** 8)

    return Recording(sensor_frame, time_frame, activities, subject,
                     name=recording_folder_path.rsplit(os.path.sep, 1)[1])


def reorder_sensor_columns(sensor_frame: pd.DataFrame) -> pd.DataFrame:
    """
    reorders according to global settings
    """

    column_suffix_dict = {}
    for column_name in sensor_frame.columns:
        ending = column_name[-2:]
        if ending in column_suffix_dict:
            column_suffix_dict[ending].append(column_name)
        else:
            column_suffix_dict[ending] = [column_name]

    # assert list(column_suffix_dict.keys()) == settings.SENSOR_SUFFIX_ORDER ... only same elements

    column_names_ordered = []
    for sensor_suffix in Config.sonar_sensor_suffix_order:
        column_names_ordered.extend(column_suffix_dict[sensor_suffix])

    return sensor_frame[column_names_ordered]
