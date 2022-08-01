# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

from gc import callbacks
import os
from abc import ABC, abstractmethod
from math import sqrt
from typing import Any, Union
from numpy.core.numeric import full
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical  # type: ignore
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore
from datetime import datetime
import os

from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore

from utils.array_operations import split_list_by_percentage, transform_to_subarrays
from utils.Recording import Recording
from utils.Window import Window

from utils.typing import assert_type
import utils.settings as settings
from utils.folder_operations import create_folders_in_path

import wandb
from wandb.keras import WandbCallback


class RainbowModel(ABC):

    # general
    model_name = None
    class_weight = None
    model: Any = None

    # Input Params
    n_features: Union[int, None] = None
    n_outputs: Union[int, None] = None
    window_size: Union[int, None] = None

    # Training Params
    batch_size: Union[int, None] = None
    n_epochs: Union[int, None] = None
    learning_rate: Union[float, None] = None

    # Config
    wandb_project: Union[str, None] = None
    verbose: Union[int, None] = 1
    kwargs = None

    # @abstractmethod
    def __init__(self, **kwargs):
        """
        Builds a model, assigns it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.
        """

        # self.model = None
        # assert (self.model is not None)
        self.window_size = kwargs.get("window_size", None)
        self.n_features = kwargs.get("n_features", None)
        self.n_outputs = kwargs.get("n_outputs", None)
        self.batch_size = kwargs.get("batch_size", None)
        self.n_epochs = kwargs.get("n_epochs", None)
        self.learning_rate = kwargs.get("learning_rate", None)
        self.validation_split = kwargs.get("validation_split", 0.2)
        self.verbose = kwargs.get("verbose", 1)
        self.class_weight = kwargs.get("class_weight", None)
        self.wandb_config = kwargs.get("wandb_config", None)

        self.kwargs = kwargs

        # Important declarations
        assert self.window_size is not None, "window_size is not set"
        assert self.n_features is not None, "n_features is not set"
        assert self.n_outputs is not None, "n_outputs is not set"
        assert self.batch_size is not None, "batch_size is not set"
        assert self.n_epochs is not None, "n_epochs is not set"
        assert self.learning_rate is not None, "learning_rate is not set"
        self.model = self._create_model(self.n_features, self.n_outputs)
        self.model.summary()

    def _create_model(self) -> tf.keras.Model:
        """
        Subclass Responsibility:
        returns a keras model
        """
        raise NotImplementedError
        
    def windowize_convert(
        self, recordings_train: "list[Recording]", should_shuffle=True
    ) -> "tuple[np.ndarray,np.ndarray]":
        """
        shuffles the windows
        """
        windows_train = self.windowize(recordings_train)
        if should_shuffle:
            shuffle(
                windows_train
            )  # many running windows in a row?, one batch too homogenous?, lets shuffle
        X_train, y_train = self.convert(windows_train)
        return X_train, y_train

    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
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
                        sensor_subarray, recording.activity, recording.subject, recording.recording_index
                    ),
                    sensor_subarrays,
                )
            )
            windows.extend(recording_windows)
        return windows

    def convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.data_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=settings.DATA_CONFIG.n_activities(),
        )

        return np.array(sensor_arrays), np.array(activity_vectors)

    # Fit ----------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the self.model to the data
        """
        assert_type(
            [(x_train, (np.ndarray, np.generic)), (y_train, (np.ndarray, np.generic))]
        )
        assert (
            x_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train have to have the same length"

        # Wandb
        callbacks = None
        if self.wandb_config is not None:
            assert (
                self.wandb_config["project"] is not None
            ), "Wandb project name is not set"
            assert (
                self.wandb_config["entity"] is not None
            ), "Wandb entity name is not set"
            assert self.wandb_config["name"] is not None, "Wandb name is not set"

            wandb.init(
                project=str(self.wandb_config["project"]),
                entity=self.wandb_config["entity"],
                name=str(self.wandb_config["name"]),
                settings=wandb.Settings(start_method='fork')
            )
            wandb.config = {
                "learning_rate": self.learning_rate,
                "epochs": self.n_epochs,
                "batch_size": self.batch_size,
            }
            callbacks = [wandb.keras.WandbCallback()]

        self.history = self.model.fit(
            x_train,
            y_train,
            validation_split=self.validation_split,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            callbacks=callbacks
        )



    # Predict ------------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)

    def export(self, path: str) -> None:
        """
        will create an 'export' folder in the path, and save the model there in 3 different formats
        """
        print("Exporting model ...")

        # Define, create folder structure
        export_path = os.path.join(path, "export")
        export_path_raw_model = os.path.join(export_path, "raw_model")
        create_folders_in_path(export_path_raw_model)

        # 1/3 Export raw model ------------------------------------------------------------
        self.model.save(export_path_raw_model)

        # 2/3 Export .h5 model ------------------------------------------------------------
        self.model.save(export_path + "/" + self.model_name + ".h5", save_format="h5")

        # 3/3 Export .h5 model ------------------------------------------------------------
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]  # Refactoring Idea: Optimizations for new tensorflow version
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        tflite_model = converter.convert()
        with open(f"{export_path}/{self.model_name}.tflite", "wb") as f:
            f.write(tflite_model)

        print("Export finished")

