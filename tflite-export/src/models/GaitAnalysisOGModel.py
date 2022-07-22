from statistics import mode
from numpy import gradient
from models.RainbowModel import RainbowModel
import tensorflow as tf
import tensorflow.keras as keras
from utils.Window import Window
from utils.Recording import Recording


class GaitAnalysisOGModel(RainbowModel):
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

        self.model_name = "GaitAnalysisTLModel"

        print(
            f"Building model for {self.window_size} timesteps (window_size) and {kwargs['n_features']} features"
        )
        self.callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
        )

    def _create_model(self):
        self.inner_model = tf.keras.Sequential(
            [tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation="relu")]
        )
        outer_model = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dropout(0.2),
            ]
        )
        inputs = tf.keras.Input(shape=(self.window_size, self.n_features))
        x = self.inner_model(inputs)
        x = outer_model(x)
        outputs = tf.keras.layers.Dense(units=self.n_outputs, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),  # "binary_crossentropy"
            metrics=self.metrics,
        )  # , precision, recall gives an error with the combined versions of keras-metrics, keras, and tf

        return model

    def _windowize_recording(self, recording: "Recording") -> "list[Window]":
        """
        :param recording:
        :return:
        """
        # windowize the recording

        windows = []
        recording_sensor_array = recording.sensor_frame.to_numpy()
        activities = recording.activity_frame.to_numpy()

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
