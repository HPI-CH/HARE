from utils.typing import assert_type

import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import Union
from utils.Window import Window


@dataclass
class Recording:
    """
    our base data object
    Multilabel, so expects activity pd.Series

    Future: 
        - a dataclass creates the intializer automatically
            - consider only giving the attributes as class vars -> dataclass handles this
        - add self.recorder

    Refactoring idea:
    - subject should be a int!, add it to assert_type
    """

    def __init__(
        self,
        sensor_frame: pd.DataFrame,
        time_frame: pd.Series,
        activities: pd.Series,
        subject: Union[str, int],
        recording_index: int,
    ) -> None:
        assert_type(
            [
                (sensor_frame, pd.DataFrame),
                (time_frame, pd.Series),
                (activities, pd.Series),
                (recording_index, int),
            ]
        )
        # TODO uncomment
        # assert isinstance(activities[0], np.float64) or isinstance(
        #    activities[0], np.int64
        # )
        assert (
            sensor_frame.shape[0] == time_frame.shape[0]
        ), "sensor_frame and time_frame have to have the same length"
        assert (
            sensor_frame.shape[0] == activities.shape[0]
        ), "sensor_frame and activities have to have the same length"

        self.sensor_frame = sensor_frame
        self.time_frame = time_frame
        self.activities = activities
        self.subject = subject
        self.recording_index = recording_index

    def windowize(self, window_size: int, features: "Union[list[str], None]" = None) -> "list[Window]":
        windows = []

        sensor_frame = self.sensor_frame if features == None else self.sensor_frame[features]

        recording_sensor_array = (
            sensor_frame.to_numpy()
        )  # recording_sensor_array[timeaxis/row, sensoraxis/column]
        activities = self.activities.to_numpy()

        start = 0
        end = 0

        def last_start_stamp_not_reached(start):
            return start + window_size - 1 < len(recording_sensor_array)

        while last_start_stamp_not_reached(start):
            end = start + window_size - 1

            # has planned window the same activity in the beginning and the end?
            if (
                len(set(activities[start: (end + 1)])) == 1
            ):  # its important that the window is small (otherwise can change back and forth) # activities[start] == activities[end] a lot faster probably
                window_sensor_array = recording_sensor_array[
                    start: (end + 1), :
                ]  # data[timeaxis/row, featureaxis/column] data[1, 2] gives specific value, a:b gives you an interval

                activity = activities[start]  # the first data point is enough
                start += (
                    window_size // 2
                )  # 50% overlap!!!!!!!!! - important for the waste calculation
                windows.append(
                    Window(
                        window_sensor_array,
                        int(activity),
                        self.subject,
                        self.recording_index,
                    )
                )

            # if the frame contains different activities or from different objects, find the next start point
            # if there is a rest smaller than the window size -> skip (window small enough?)
            else:
                # find the switch point -> start + 1 will than be the new start point
                # Refactoring idea (speed): Have switching point array to find point immediately
                # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy/19125898#19125898
                while last_start_stamp_not_reached(start):
                    if activities[start] != activities[start + 1]:
                        start += 1
                        break
                    start += 1
        return windows

    def split_by_percentage(self, test_percentage: float) -> "list[Recording]":
        """
        splits the recording into two recordings, the first one is the training set, the second one is the test set
        """
        assert_type([(test_percentage, float)])
        assert 0 <= test_percentage <= 1, "test_percentage has to be between 0 and 1"

        # find index to split at
        split_index = int(len(self.activities) * (1 - test_percentage))

        # gather test data
        test_sensor_frame = self.sensor_frame.iloc[split_index:, :]
        test_time_frame = self.time_frame.iloc[split_index:]
        test_activities = self.activities.iloc[split_index:]
        # reindex test data
        test_sensor_frame.index = range(len(self.activities) - split_index)
        test_time_frame.index = range(len(self.activities) - split_index)
        test_activities.index = range(len(self.activities) - split_index)

        # create test recordings
        test_recording = Recording(
            sensor_frame=test_sensor_frame,
            time_frame=test_time_frame,
            activities=test_activities,
            subject=self.subject,
            recording_index=self.recording_index,
        )

        # gather training data
        train_sensor_frame = self.sensor_frame.iloc[:split_index, :]
        train_time_frame = self.time_frame.iloc[:split_index]
        train_activities = self.activities.iloc[:split_index]

        # reindex training data
        train_sensor_frame.index = range(split_index)
        train_time_frame.index = range(split_index)
        train_activities.index = range(split_index)

        # create training recording
        train_recording = Recording(
            sensor_frame=train_sensor_frame,
            time_frame=train_time_frame,
            activities=train_activities,
            subject=self.subject,
            recording_index=self.recording_index,
        )

        return [train_recording, test_recording]

    def split_by_indexes(self, index: list) -> list:
        """
        splits the recording into two recordings, the first one is the training set, the second one is the test set
        """
        assert_type([(index, list)])
        assert len(index) == 2, "index has to be a list of two indices"
        assert index[0] < index[1], "index[0] has to be smaller than index[1]"

        # gather test data
        test_sensor_frame = self.sensor_frame.iloc[index[0]:index[1], :]
        test_time_frame = self.time_frame.iloc[index[0]:index[1]:]
        test_activities = self.activities.iloc[index[0]:index[1]:]
        # reindex test data
        test_sensor_frame.index = range((index[1] - index[0]))
        test_time_frame.index = range((index[1] - index[0]))
        test_activities.index = range((index[1] - index[0]))

        # create test recordings
        test_recording = Recording(
            sensor_frame=test_sensor_frame,
            time_frame=test_time_frame,
            activities=test_activities,
            subject=self.subject,
            recording_index=self.recording_index,
        )
        if index[0] == 0:
            first_train_recording = None
        else:

            # gather training data
            first_train_sensor_frame = self.sensor_frame.iloc[:index[0], :]
            first_train_time_frame = self.time_frame.iloc[:index[0]:]
            first_train_activities = self.activities.iloc[:index[0]:]

            # reindex training data
            first_train_sensor_frame.index = range(index[0])
            first_train_time_frame.index = range(index[0])
            first_train_activities.index = range(index[0])

            # create first training recording
            first_train_recording = Recording(
                sensor_frame=first_train_sensor_frame,
                time_frame=first_train_time_frame,
                activities=first_train_activities,
                subject=self.subject,
                recording_index=self.recording_index,
            )
        if index[1] == len(self.activities):
            second_train_recording = None
        else:
            # gather second training data
            print(self.sensor_frame.shape)
            print(self.time_frame.shape)
            print(self.activities.shape)
            second_train_sensor_frame = self.sensor_frame.iloc[index[1]:, :]
            second_train_time_frame = self.time_frame.iloc[index[1]:]
            second_train_activities = self.activities.iloc[index[1]:]
            print(index)
            print(second_train_sensor_frame.shape)
            print(len(self.activities) - index[1])
            # reindex training data
            second_train_sensor_frame.index = range(
                len(self.sensor_frame) - index[1])
            second_train_time_frame.index = range(
                len(self.time_frame) - index[1])
            second_train_activities.index = range(
                len(self.activities) - index[1])

            # create second training recording
            second_train_recording = Recording(
                sensor_frame=second_train_sensor_frame,
                time_frame=second_train_time_frame,
                activities=second_train_activities,
                subject=self.subject,
                recording_index=self.recording_index,
            )

        return [first_train_recording, second_train_recording, test_recording]
