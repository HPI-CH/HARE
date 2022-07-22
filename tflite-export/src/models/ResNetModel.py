# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value
# some imports are not accepted by pylint

from models.RainbowModel import RainbowModel
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from utils.typing import assert_type
from utils.Window import Window
from utils.Recording import Recording
import itertools


class ResNetModel(RainbowModel):
    def __init__(self, **kwargs):
        """

        epochs=10
        :param kwargs:
            window_size: int
            n_features: int
            n_outputs: int
        """

        # hyper params to instance vars
        super().__init__(**kwargs)

        self.model_name = "resnet_model"

        # create model

        # Refactoring idea:
        # n_features of a neuronal net is the number of inputs, so in reality n_features = window_size * n_features
        # we could have another name for that

        print(
            f"Building model for {self.window_size} timesteps (window_size) and {kwargs['n_features']} features"
        )
        self.callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
        )

    def _create_model(self):
        n_feature_maps = 64

        input_layer = keras.layers.Input((self.window_size, self.n_features))
        x = self._preprocessing_layer(input_layer)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=8, padding="same"
        )(x)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps, kernel_size=1, padding="same"
        )(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation("relu")(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=1, padding="same"
        )(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation("relu")(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=8, padding="same"
        )(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation("relu")(conv_x)

        conv_y = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=5, padding="same"
        )(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation("relu")(conv_y)

        conv_z = keras.layers.Conv1D(
            filters=n_feature_maps * 2, kernel_size=3, padding="same"
        )(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation("relu")(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(self.n_outputs, activation="softmax")(
            gap_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(),
            metrics=self.metrics,
        )

        return model

    def _windowize_recording(self, recording: "Recording") -> "list[Window]":
        windows = []
        recording_sensor_array = recording.sensor_frame.to_numpy()
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
