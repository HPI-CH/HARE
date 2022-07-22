import os
import json
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from utils.data_set import DataSet

from utils.file_functions import get_subfolder_names
from utils import settings
from utils.Recording import Recording
from loader.XSensRecordingReader import XSensRecordingReader


def initialize_dataconfig(data_config):
    settings.init(data_config)


def load_sonar_dataset(
    dataset_path: str,
    limit_n_recs: int = None,
    multiprocessing: bool = True,
    data_config=None,
) -> "list[Recording]":
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
    if multiprocessing:
        assert (
            data_config is not None
        ), "data config needs to be accessible, you need to do settings.init in the runner.py"

    if not os.path.exists(dataset_path):
        raise Exception("The dataset_path does not exist")

    recordings: "list[Recording]" = []

    # recording
    recording_folder_names = get_subfolder_names(dataset_path)

    recording_folder_names = [
        os.path.join(dataset_path, recording_folder_name)
        for recording_folder_name in recording_folder_names
    ]

    if limit_n_recs is not None:
        enumerated_recording_folder_names = list(
            enumerate(recording_folder_names[:limit_n_recs])
        )
    else:
        enumerated_recording_folder_names = list(enumerate(recording_folder_names))

    # USE ONE (Multiprocessing or Single Thread)
    # Multiprocessing:
    if multiprocessing:
        pool = Pool(10, initializer=initialize_dataconfig, initargs=(data_config,))
        recordings = pool.imap_unordered(
            read_recording_from_folder, enumerated_recording_folder_names, 10
        )
        pool.close()
        pool.join()

    # Single thread:
    else:
        recordings = [
            read_recording_from_folder(recording_folder_name, continue_on_error=False)
            for recording_folder_name in enumerated_recording_folder_names
        ]

    assert len(recordings) > 0, "load_sonar_dataset: recordings empty!"
    return recordings


def read_recording_from_folder(
    enumerated_recording_folder_names: "tuple(int, str)", continue_on_error: bool = True
):
    recording_folder_path = enumerated_recording_folder_names[1]
    recording_idx = enumerated_recording_folder_names[0]
    try:
        subject_folder_name = get_subject_folder_name(recording_folder_path)
        return create_recording(
            recording_folder_path, subject_folder_name, recording_idx
        )
    except Exception as e:
        if continue_on_error:
            print(
                "===> Will skip Recording, because error while reading! path:"
                + recording_folder_path
                + "\nError:\n\t"
                + str(e)
            )
            return None
        raise e


def get_subject_folder_name(recording_folder_path: str) -> str:
    with open(
        recording_folder_path + os.path.sep + "metadata.json", "r", encoding="utf8"
    ) as f:
        data = json.load(f)
    return data["person"]


def get_activity_dataframe(time_frame, recording_folder_path: str) -> pd.DataFrame:
    with open(
        recording_folder_path + os.path.sep + "metadata.json", "r", encoding="utf8"
    ) as f:
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

    def str_label_to_activity_idx(label_obj: dict):
        label_obj["label"] = settings.DATA_CONFIG.raw_label_to_activity_idx(
            label_obj["label"]
        )
        return label_obj

    activities_meta = list(map(label_timestamp_to_microseconds, activities_meta))
    activities_meta = list(map(str_label_to_activity_idx, activities_meta))

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

    assert activities_per_timestep.shape[0] == pd_time_frame.shape[0]
    return activities_per_timestep


def create_recording(
    recording_folder_path: str, subject: str, recording_idx: int
) -> Recording:
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

    return Recording(
        sensor_frame=sensor_frame,
        time_frame=time_frame,
        activities=activity,
        subject=settings.DATA_CONFIG.raw_subject_to_subject_idx(subject),
        recording_index=recording_idx
    )


def reorder_sensor_columns(
    rec_folder_path: str, sensor_frame: pd.DataFrame
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

    # assert list(column_suffix_dict.keys()) == settings.DATA_CONFIG.sensor_suffix_order ... only same elements

    # Catch errors and output the file where it goes wrong
    try:
        column_names_ordered = []
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            column_names_ordered.extend(column_suffix_dict[sensor_suffix])

        return sensor_frame[column_names_ordered]
    except KeyError:
        print(f"Could not find sensor suffixes in {rec_folder_path}")
        return None
