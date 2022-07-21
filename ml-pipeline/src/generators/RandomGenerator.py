from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from loading.cache_recording import read_recording_from_csv
from sonar_types import Recording, Window, PreprocessingFunction


class DynamicActivityDict:
    activities_str_to_id: list[str] = []

    def get(self, val: Union[str, int]):
        if isinstance(val, str):
            return self._get_str(val)
        else:
            return self._get_int(val)

    def _get_str(self, val: str):
        return self.activities_str_to_id.index(val)

    def _get_int(self, val: int):
        return self.activities_str_to_id[val]

    def add_from_labels(self, labels: pd.Series):
        activities = set(labels)
        for activity in activities:
            if activity not in self.activities_str_to_id:
                self.activities_str_to_id.append(activity)

    def as_dict(self):
        # e.g. {'null': 0, 'walk': 1, ...}
        return {v: i for i, v in enumerate(self.activities_str_to_id)}


@dataclass
class LoadedFile:
    windows: list
    file_path: str


PreprocessorFnT = Callable[[Recording], Recording]
WindowFnT = Callable[[Recording], list[Window]]
ModelConverterT = Callable[[Window], tuple[np.ndarray, int]]


class RecordingLoader:
    """
    Saves RAM by using multiple randomly selected files at the same time to present random windows.

    recordings_to_load are shuffled as well as windows, when loaded
    """
    n_files: int
    _preprocessor: PreprocessorFnT
    _window_converter: WindowFnT
    _model_converter: ModelConverterT

    # The recordings to load in the current iteration
    _recordings_to_load: list[str]

    # The recordings that are / have been available to the generator
    _recordings: list[str]
    _loaded_files: list[LoadedFile] = []
    _shuffle_windows: bool

    def __init__(self,
                 *,
                 recordings_to_load: list[str],
                 preprocessor: PreprocessorFnT,
                 window_converter: WindowFnT,
                 model_converter: ModelConverterT,
                 n_files: int = 5,
                 shuffle_recordings: bool = True,
                 shuffle_windows: bool = True,
                 ):
        _recordings = copy.deepcopy(recordings_to_load)
        if shuffle_recordings:
            random.seed(42)
            random.shuffle(recordings_to_load, )

        self._recordings = recordings_to_load
        self._recordings_to_load = copy.deepcopy(_recordings)
        self._preprocessor = preprocessor
        self._window_converter = window_converter
        self._model_converter = model_converter
        self.n_files = n_files
        self._shuffle_windows = shuffle_windows

    def generator(self):
        self._recordings_to_load = copy.deepcopy(self._recordings)
        self._ensure_enough_files_loaded()
        while len(self._loaded_files) > 0:
            # Yield a random window
            # This is no uniform distribution over windows, but recordings / files!
            selected_file = random.choice(self._loaded_files) if self._shuffle_windows else self._loaded_files[0]
            selected_window = selected_file.windows.pop()
            yield self._model_converter(selected_window)

            self._ensure_enough_files_loaded()

    def _ensure_enough_files_loaded(self):
        # Filter empty files
        self._loaded_files = list(filter(lambda f: len(f.windows) > 0, self._loaded_files))

        # Load new file(s) if necessary
        while len(self._loaded_files) <= self.n_files and len(self._recordings_to_load) > 0:
            self._load_random_file()
            # Filter empty files (ensure new window is not empty)
            self._loaded_files = list(filter(lambda f: len(f.windows) > 0, self._loaded_files))

    def _load_random_file(self):
        recording_path = self._recordings_to_load.pop()
        # log(f"Loading {recording_path}")
        raw_recording = read_recording_from_csv(recording_path)
        recording = self._preprocessor(raw_recording)
        if recording is None:
            return
        windows = self._window_converter(recording)
        if self._shuffle_windows:
            random.shuffle(windows)
        self._loaded_files.append(LoadedFile(windows=windows, file_path=recording_path))

    def as_dataset(self) -> tf.data.Dataset:
        self._ensure_enough_files_loaded()

        # Get one representative to type dataset
        X_repr, y_repr = self._model_converter(self._loaded_files[0].windows[0])
        window_typespec = tf.type_spec_from_value(X_repr)
        label_typespec = tf.type_spec_from_value(y_repr)

        return tf.data.Dataset.from_generator(
            self.generator,
            output_signature=(
                window_typespec,
                label_typespec,
            ),
        )


def map_activities_to_int_async(*, activity_map: DynamicActivityDict) -> PreprocessingFunction:
    def fn(recordings: list[Recording]):
        assert len(recordings) == 1, 'this function currently only works with a single recording per call'
        recording = recordings[0]
        activity_map.add_from_labels(recording.activities)
        recording.activities.replace(activity_map.as_dict(), inplace=True)
        return [recording]

    return fn
