import json
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas import Series

from loading.XSensRecordingReader import XSensRecordingReader
from sonar_types import Recording
from utils.config import Config
from utils.path_helpers import get_subfolder_names


def load_dataset(dataset_path: str, limit: int = None) -> list[Recording]:
    """
    Returns a list of the raw recordings (activities, subjects included, None values) (different representaion of dataset)
    directory structure bias! not shuffled!


    This function knows the structure of the XSens dataset.
    It will call the create_recording_frame function on every recording folder.

    bp/data
        dataset_01
            recording_01
                metadata.json
                sensor_01.csv
                sensor_02.csv

            recording_01
                metadata.json
                sensor_01.csv
                sensor_02.csv

        data_set_02
            ...

    """

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: list[Recording] = []

    # recording
    recording_folder_names = get_subfolder_names(dataset_path)
    recording_folder_names = [
        os.path.join(dataset_path, recording_folder_name)
        for recording_folder_name in recording_folder_names
    ]

    if limit is not None:
        recording_folder_names = recording_folder_names[:limit]

    # USE ONE (Multiprocessing or Single Thread)
    # Multiprocessing:
    pool = Pool()
    recordings = pool.imap_unordered(
        read_recording_from_folder, recording_folder_names, 10
    )
    pool.close()
    pool.join()

    # Single thread:
    # recordings = [read_recording_from_folder(recording_folder_name) for recording_folder_name in recording_folder_names]

    return list(filter(lambda x: x is not None, recordings))


def read_recording_from_folder(recording_folder_path: str) -> 'Recording':
    try:
        subject_folder_name = get_subject_folder_name(recording_folder_path)
        return create_recording(recording_folder_path, subject_folder_name)
    except Exception as e:
        print("Error while reading recording from folder: " + recording_folder_path)
        print(e)
        print(traceback.format_exc())
        return None


def get_subject_folder_name(recording_folder_path: str) -> str:
    with open(os.path.join(recording_folder_path, "metadata.json"), "r") as f:
        data = json.load(f)
    return data["person"]


def get_activity_dataframe(time_frame, recording_folder_path: str) -> Series:
    with open(os.path.join(recording_folder_path, "metadata.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    # The activities as a list of objects with label & timeStarted
    activities_meta = data["activities"]
    pd_time_frame = time_frame.to_frame()
    # the objective - the activities as series, synchronized with the recording sensor_frame
    activities_per_timestep = pd.Series(np.empty(shape=(pd_time_frame.shape[0])))

    # Convert all timestamps to microseconds and a number
    def timestamp_to_microseconds(timestamp: str):
        return int(timestamp) * 1000

    def label_timestamp_to_microseconds(label_obj: dict):
        label_obj["timeStarted"] = timestamp_to_microseconds(label_obj["timeStarted"])
        return label_obj

    activities_meta = list(map(label_timestamp_to_microseconds, activities_meta))

    # Now we have all timesteps in the same format (microseconds), but the label timestamps are still offset by some value
    # To fix / work around that, we always take the duration of one label and add it to the labels current SampleTimeFine
    # using the last labels end time repeatedly

    # As the end timestamp is always excluded, we add a bit 1 to the last timestamp to get all lines
    end_timestamp = timestamp_to_microseconds(data["endTimestamp"]) + 100
    sampletime_start = pd_time_frame.iloc[0]["SampleTimeFine"]

    sampletime_activity_list = []
    for i in range(len(activities_meta)):
        current_activity_timestamp = activities_meta[i]["timeStarted"]
        next_activity_timestamp = (
            activities_meta[i + 1]["timeStarted"]
            if i < len(activities_meta) - 1
            else end_timestamp
        )

        start = sampletime_start if i == 0 else sampletime_activity_list[-1][1]
        end = start + (next_activity_timestamp - current_activity_timestamp)

        sampletime_activity_list.append((start, end))

    # Now we have a list of tupels which give the time and end SampleTimeFine of all activities
    # We can now assign the activities to the timesteps by the cool pandas / numpy functions :)

    for idx, (start, end) in enumerate(sampletime_activity_list):
        # First find all indices where SampleTimeFine is in between start and end
        matching_indices = np.where(
            (pd_time_frame["SampleTimeFine"] >= start)
            & (pd_time_frame["SampleTimeFine"] < end)
        )
        # Now write the label for all these indices
        activities_per_timestep.iloc[matching_indices] = activities_meta[idx]["label"]

    # activity_to_id_map = {
    #    v: k for k, v in enumerate(Config.sonar_labels)
    # }
    # activities_per_timestep = activities_per_timestep.map(activity_to_id_map)
    assert activities_per_timestep.shape[0] == pd_time_frame.shape[0]
    return activities_per_timestep


def create_recording(recording_folder_path: str, subject: str) -> Recording:
    """
    Returns a recording
    Gets a XSens recorind folder path, loops over sensor files, concatenates them, adds activity and subject, returns a recording
    """

    print(recording_folder_path)
    time1 = time.time()
    raw_recording_frame = XSensRecordingReader.get_recording_frame(
        recording_folder_path
    )

    # If less than a second is recorded or
    if raw_recording_frame.shape[0] < 60:
        print(f"Dropping because less than 60 steps: {recording_folder_path}")
        return
    time2 = time.time()
    time_column_name = "SampleTimeFine"
    time_frame = raw_recording_frame[time_column_name]

    activity = get_activity_dataframe(time_frame, recording_folder_path)
    time3 = time.time()

    sensor_frame = raw_recording_frame.drop([time_column_name], axis=1)
    sensor_frame = reorder_sensor_columns(recording_folder_path, sensor_frame)
    time4 = time.time()
    print(
        "took: {:.3f} - {:.3f} - {:.3f}".format(
            (time2 - time1) * 1000.0, (time3 - time2) * 1000.0, (time4 - time3) * 1000.0
        )
    )

    if sensor_frame is None:
        return None
    # subject_idx = Config.sonar_people.index(subject)
    return Recording(sensor_frame, time_frame, activities=activity, subject=subject)


def reorder_sensor_columns(
        rec_folder_path: str, sensor_frame: pd.DataFrame, suffix_order=Config.sonar_sensor_suffix_order
) -> pd.DataFrame:
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

    # Catch errors and output the file where it goes wrong
    try:
        column_names_ordered = []
        for sensor_suffix in suffix_order:
            column_names_ordered.extend(column_suffix_dict[sensor_suffix])

        return sensor_frame[column_names_ordered]
    except KeyError:
        print(f"Could not find sensor suffixes in {rec_folder_path}")
        return None
