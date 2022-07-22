import re
from xmlrpc.client import Boolean
from utils.array_operations import split_list_by_percentage
from utils.typing import assert_type
from utils.Recording import Recording
from utils.Window import Window
import numpy as np
from utils.typing import assert_type
import itertools
from tensorflow.keras.utils import to_categorical
from scipy import signal
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Union


class DataSet(list):
    def __init__(self, data: "Union[list[Recording], DataSet]" = None, data_config=None):
        if not data is None:
            self.extend(data)
            if isinstance(data, DataSet):
                self.data_config = data.data_config
            else:
                assert data_config != None, "You have passed data as a list of recordings. In this case you must also pass a data_config which is not None"
                self.data_config = data_config
        else:
            assert data_config != None, "You have not passed any data to this data set. In this case you must pass a data_config which is not None"
            self.data_config = data_config

    def windowize(self, window_size: int) -> "list[Window]":
        """
        Jens version of windowize
        - no stride size default overlapping 50 percent
        - there is is a test for that method
        - window needs be small, otherwise there will be much data loss
        """
        assert_type([(self[0], Recording)])
        assert (
            window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        if window_size > 25:
            print(
                "\n===> WARNING: the window_size is big with the used windowize algorithm (Jens) you have much data loss!!! (each activity can only be a multiple of the half the window_size, with overlapping a half of a window is cutted)\n"
            )

        self._print_jens_windowize_monitoring(window_size)
        # Refactoring idea (speed): Mulitprocessing https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop/20192251#20192251
        print("windowizing in progress ....")
        recording_windows = list(
            map(lambda recording: recording.windowize(window_size), self)
        )
        print("windowizing done")
        return list(
            itertools.chain.from_iterable(recording_windows)
        )  # flatten (reduce dimension)

    def split_leave_subject_out(self, test_subject) -> "tuple[DataSet, DataSet]":
        recordings_train = list(
            filter(lambda recording: recording.subject != test_subject, self)
        )
        recordings_test = list(
            filter(lambda recording: recording.subject == test_subject, self)
        )
        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def split_by_subjects(self, subjectsForListA: "list[str]") -> "tuple[DataSet, DataSet]":
        """
        Splits the recordings into a tuple of
            - a sublist of recordings of subjects in subjectsForListA
            - the recordings of the subjects not in subjectsForListA
        """
        a = list(
            filter(lambda recording: recording.subject in subjectsForListA, self))
        b = list(
            filter(lambda recording: recording.subject not in subjectsForListA, self))
        return DataSet(a, self.data_config), DataSet(b, self.data_config)

    def count_activities_per_subject(self) -> "pd.DataFrame":
        def activity_counts_from_rec(rec): return rec.activities.copy().map(
            lambda x: self.data_config.activity_idx_to_display_name(x))

        def subject_to_id(
            subject_label): return f"Subject {self.data_config.raw_subject_to_subject_idx(subject_label)}"

        values = pd.DataFrame(
            {subject_to_id(self[0].subject): activity_counts_from_rec(self[0]).value_counts()})
        for rec in self[1:]:
            # print(rec.activities.value_counts())

            values = values.add(pd.DataFrame(
                {subject_to_id(rec.subject): activity_counts_from_rec(rec).value_counts()}), fill_value=0)
        return values

    def count_activities_per_subject_as_dict(self) -> "dict[str, int]":
        resultDict = {}
        for recording in self:
            counts = recording.activities.value_counts()
            for activity_id, count in counts.items():
                if activity_id in resultDict:
                    resultDict[activity_id] += count
                else:
                    resultDict[activity_id] = count
        for activity in self.data_config.raw_label_to_activity_idx_map:
            if not activity in resultDict:
                resultDict[activity] = 0
        return resultDict

    def count_recordings_of_subjects(self) -> "dict[str, int]":
        subjectCount = {}
        for recording in self:
            if recording.subject in subjectCount:
                subjectCount[recording.subject] += 1
            else:
                subjectCount[recording.subject] = 1
        return subjectCount

    def get_people_in_recordings(self) -> "list[str]":
        people = set()
        for recording in self:
            people.add(recording.subject)
        return list(people)

    def plot_activities_per_subject(self, dirPath, fileName: str, title: str = ""):
        values = self.count_activities_per_subject()

        plt.figure()
        plt.rcParams.update({'font.size': 16})

        ax = values.plot.bar(figsize=(22, 25), linewidth=4, fontsize=18)

        plt.title(title, fontdict={'fontsize': 35})
        plt.legend(fontsize=20)
        plt.xlabel("Activities (Subjects)", fontsize=22)
        plt.ylabel("Measurement timestamps", fontsize=22)

        plt.savefig(os.path.join(dirPath, fileName))

    def split_by_percentage(self, test_percentage: float, intra: Boolean = False) -> "tuple[DataSet, DataSet]":
        if intra:  # TODO: check for the number of classes and split for each of the class recordings individually

            recordings_test = []
            recordings_train = []
            for recording in self:
                recording_train, recording_test = recording.split_by_percentage(
                    test_percentage)
                recordings_train.append(recording_train)
                recordings_test.append(recording_test)
        else:
            recordings_train, recordings_test = split_list_by_percentage(
                list_to_split=self, percentage_to_split=test_percentage
            )
        print(
            f"amount of recordings_train: {len(recordings_train)}\n amount of recordings_test: {len(recordings_test)}")
        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def intra_kfold_split(self, k):
        """
        Splits each of the recordings into k folds and returns the indexes of each split.
        """
        assert_type([(self[0], Recording)])
        split_indexes = [[] for elem in range(k)]
        for i in range(k):
            for recording in self:
                split_indexes[i].append(
                    [(recording.sensor_frame.shape[0]*i)//k, (recording.sensor_frame.shape[0]*(i+1))//k])
        return split_indexes

    def intra_split_by_indexes(self, indexes):
        recordings_train = []
        recordings_test = []
        print(indexes)
        for idx, recording in enumerate(self):
            first_recording_train, second_recording_train, recording_test = recording.split_by_indexes(
                indexes[idx])
            if first_recording_train is not None:
                recordings_train.append(first_recording_train)
            if second_recording_train is not None:
                recordings_train.append(second_recording_train)
            recordings_test.append(recording_test)

        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def split_by_indexes(self, train_indexes):
        assert_type([(self[0], Recording)])
        recordings_train = []
        recordings_test = []

        for idx, recording in enumerate(self):
            if idx in train_indexes:
                recordings_train.append(recording)
            else:
                recordings_test.append(recording)

        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def convert_windows_sonar(
        windows: "list[Window]", num_classes: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=num_classes,
        )

        return np.array(sensor_arrays), np.array(activity_vectors)

    def convert_windows_jens(
        windows: "list[Window]",
        num_classes: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        X_train, y_train = DataSet.convert_windows_sonar(windows, num_classes)
        return np.expand_dims(X_train, -1), y_train

    def _print_jens_windowize_monitoring(self, window_size):
        def n_wasted_timesteps_jens_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
            # (overlapping amount self.window_size // 2 from the algorithm!)

            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - window_size) % (window_size // 2)
                    if label_len >= window_size
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
            map(lambda recording: len(recording.activities), self))
        n_wasted_timesteps = sum(map(n_wasted_timesteps_jens_windowize, self))
        print(
            f"=> jens_windowize_monitoring (total recording time)\n\tbefore: {to_hours_str(n_total_timesteps)}\n\tafter: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")

    def replaceNaN_ffill(self):
        """
        the recordings have None values, this function replaces them with the last non-NaN value of the feature
        """
        assert_type([(self[0], Recording)])
        fill_method = "ffill"
        for recording in self:
            recording.sensor_frame = recording.sensor_frame.fillna(
                method=fill_method)
            recording.sensor_frame = recording.sensor_frame.fillna(
                0)

    def resample(self, target_sampling_rate: int, base_sampling_rate: int, show_plot: Boolean = False, path: str = None):
        """
        resamples the recordings to the target sampling rate
        """
        assert_type([(self[0], Recording)])
        if show_plot == True:
            x = np.linspace(0, 1, base_sampling_rate)
            y = np.array(self[0].sensor_frame.iloc[:, 0])[:base_sampling_rate]

            x_time = np.linspace(0, 1, base_sampling_rate//10)
            y_time = np.array(self[0].time_frame)[:base_sampling_rate//10]

            x_activities = np.linspace(0, 1, base_sampling_rate//10)
            y_activities = np.array(self[0].activities)[
                :base_sampling_rate//10]

        for recording in self:
            sensor_frame_resampled = pd.DataFrame(signal.resample_poly(
                x=recording.sensor_frame,
                up=target_sampling_rate,
                down=base_sampling_rate
            ))
            time_frame_resampled = pd.Series(
                list(
                    map(lambda x: x * (base_sampling_rate /
                        target_sampling_rate), recording.time_frame)
                )[:int(len(recording.time_frame) * (target_sampling_rate / base_sampling_rate))]
            )
            activities_resampled = pd.Series(
                list(map(lambda y: recording.activities[y], (list(
                    map(lambda x: int(x * (target_sampling_rate /
                        base_sampling_rate)), recording.activities.index)
                )[:int(len(recording.activities.index) * (target_sampling_rate / base_sampling_rate))])))
            )

            # make sure the length of the resampled data is equal
            recording.sensor_frame = sensor_frame_resampled
            recording.time_frame = time_frame_resampled
            recording.activities = activities_resampled
            min_length = min(len(recording.sensor_frame), len(
                recording.time_frame), len(recording.activities))
            recording.sensor_frame = recording.sensor_frame.iloc[:min_length]
            recording.time_frame = recording.time_frame.iloc[:min_length]
            recording.activities = recording.activities.iloc[:min_length]

        if show_plot == True:
            x_new = np.linspace(0, 1, target_sampling_rate)
            y_new = np.array(self[0].sensor_frame.iloc[:, 0])[
                :target_sampling_rate]

            x_time_new = np.linspace(0, 1, target_sampling_rate//10)
            y_time_new = np.array(self[0].time_frame)[
                :target_sampling_rate//10]

            x_activities_new = np.linspace(0, 1, target_sampling_rate//10)
            y_activities_new = np.array(self[0].activities)[
                :target_sampling_rate//10]
            plt.plot(x, y, 'g.-', x_new, y_new, 'r.-', 1)
            plt.legend(['data', 'resampled'], loc='best')

            if path != None:
                plt.savefig(os.path.join(path, 'resampling.png'))
            else:
                plt.savefig('resampling.png')
            plt.show()
            plt.close()
            plt.plot(x_activities, y_activities, 'g.-',
                     x_activities_new, y_activities_new, 'r.-', 1)
            plt.legend(['data', 'resampled'], loc='best')

            if path != None:
                plt.savefig(os.path.join(path, 'resampling_activities.png'))
            else:
                plt.savefig('resampling_activities.png')
            plt.show()
            plt.close()
            plt.plot(x_time, y_time, 'g.-', x_time_new, y_time_new, 'r.-', 1)
            plt.legend(['data', 'resampled'], loc='best')

            if path != None:
                plt.savefig(os.path.join(path, 'resampling_time.png'))
            else:
                plt.savefig('resampling_time.png')
            plt.show()
            plt.close()
