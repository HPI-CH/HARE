from utils.typing import assert_type
from utils.Recording import Recording
from utils.Window import Window
from utils.array_operations import transform_to_subarrays
from typing import Union
import itertools
import numpy as np
import os
import pandas as pd
import utils.settings as settings


class Windowizer:
    stride_size: Union[int, None] = None

    def __init__(self, window_size):
        self.window_size = window_size

    def sonar_windowize(self, recordings: "list[Recording]") -> "list[Window]":
        """
        based on the hyper param for window size, windowizes the recording_frames
        convertion to numpy arrays
        """
        assert_type([(recordings[0], Recording)])

        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        assert (
            self.stride_size is not None
        ), "stride_size has to be set in the constructor of your concrete model class, please"

        windows: "list[Window]" = []
        for recording in recordings:
            sensor_array = recording.sensor_frame.to_numpy()
            sensor_subarrays = transform_to_subarrays(
                sensor_array, self.window_size, self.stride_size
            )
            recording_windows = list(
                map(
                    lambda sensor_subarray: Window(
                        sensor_subarray,
                        recording.activity,
                        recording.subject,
                        recording.recording_index,
                    ),
                    sensor_subarrays,
                )
            )
            windows.extend(recording_windows)
        return windows

    def jens_windowize(self, recordings: "list[Recording]") -> "list[Window]":
        """
        Jens version of windowize
        - no stride size default overlapping 50 percent
        - there is is a test for that method
        - window needs be small, otherwise there will be much data loss
        """
        assert_type([(recordings[0], Recording)])
        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        if self.window_size > 25:
            print(
                "\n===> WARNING: the window_size is big with the used windowize algorithm (Jens) you have much data loss!!! (each activity can only be a multiple of the half the window_size, with overlapping a half of a window is cutted)\n"
            )

        self._print_jens_windowize_monitoring(recordings)
        # Refactoring idea (speed): Mulitprocessing https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop/20192251#20192251
        print("windowizing in progress ....")
        recording_windows = list(map(self._windowize_recording, recordings))
        print("windowizing done")
        return list(
            itertools.chain.from_iterable(recording_windows)
        )  # flatten (reduce dimension)

    # Helpers ---------------------------------------------------------------------------------------------------------

    def _windowize_recording(self, recording: "Recording") -> "list[Window]":
        windows = []
        recording_sensor_array = (
            recording.sensor_frame.to_numpy()
        )  # recording_sensor_array[timeaxis/row, sensoraxis/column]
        activities = recording.activities.to_numpy()

        start = 0
        end = 0

        def last_start_stamp_not_reached(start):
            return start + self.window_size - 1 < len(recording_sensor_array)

        while last_start_stamp_not_reached(start):
            end = start + self.window_size - 1

            # has planned window the same activity in the beginning and the end?
            if (
                len(set(activities[start : (end + 1)])) == 1
            ):  # its important that the window is small (otherwise can change back and forth) # activities[start] == activities[end] a lot faster probably
                window_sensor_array = recording_sensor_array[
                    start : (end + 1), :
                ]  # data[timeaxis/row, featureaxis/column] data[1, 2] gives specific value, a:b gives you an interval
                activity = activities[start]  # the first data point is enough
                start += (
                    self.window_size // 2
                )  # 50% overlap!!!!!!!!! - important for the waste calculation
                windows.append(
                    Window(
                        window_sensor_array,
                        int(activity),
                        recording.subject,
                        recording.recording_index,
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

    def _print_jens_windowize_monitoring(self, recordings: "list[Recording]"):
        def n_wasted_timesteps_jens_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
            # (overlapping amount self.window_size // 2 from the algorithm!)
            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - self.window_size) % (self.window_size // 2)
                    if label_len >= self.window_size
                    else label_len
                )

            # Refactoring to map? Would need an array lookup per change_idx (not faster?!)
            start_idx = 0
            n_wasted_timesteps = 0
            for change_idx in change_idxs:
                label_len = change_idx - start_idx
                n_wasted_timesteps += get_n_wasted_timesteps(label_len)
                start_idx = change_idx
            last_label_len = (
                len(activities) - change_idxs[-1]
                if len(change_idxs) > 0
                else len(activities)
            )
            n_wasted_timesteps += get_n_wasted_timesteps(last_label_len)
            return n_wasted_timesteps

        def to_hours_str(n_timesteps) -> int:
            hz = 30
            minutes = (n_timesteps / hz) / 60
            hours = int(minutes / 60)
            minutes_remaining = int(minutes % 60)
            return f"{hours}h {minutes_remaining}m"

        n_total_timesteps = sum(
            map(lambda recording: len(recording.activities), recordings)
        )
        n_wasted_timesteps = sum(map(n_wasted_timesteps_jens_windowize, recordings))
        print(
            f"=> jens_windowize_monitoring (total recording time)\n\tbefore: {to_hours_str(n_total_timesteps)}\n\tafter: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")

