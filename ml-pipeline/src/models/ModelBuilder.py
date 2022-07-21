from __future__ import annotations

from abc import ABC
from dataclasses import field
from typing import Union, Callable

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

from sonar_types import Window


class ModelBuilder(ABC):
    """
    Class to store all configuration and one built model for a specific model type.
    """
    # general
    # model_name: str
    model_builder: Callable[[ModelBuilder], Model] = field(repr=False)
    n_outputs: int

    # Variables with a default
    class_weight = None
    # learning_rate = 0.0001

    model: Model = field(repr=False)
    verbose: Union[int, None] = None

    def build(self) -> Model:
        """
        Builds a model given the model_builder. The builder gets the ModelBuilder self delegated to prevent
        all the inheritance overhead.
        """

        # Self is passed automatically, so no need to pass it (otherwise error occurs)
        self.model = self.model_builder()  # type: ignore
        return self.model


def split_data_and_labels(builder: ModelBuilder, windows: list[Window]):
    """
    Splits the windows into sensor data and activities
    """
    sensor_arrays = list(map(lambda window: window.sensor_frame, windows))
    activities = list(map(lambda window: window.activity, windows))

    # to_categorical converts the activity_array to the dimensions needed
    activity_vectors = to_categorical(
        np.array(activities),
        num_classes=builder.n_outputs,
    )

    return np.array(sensor_arrays), np.array(activity_vectors)
