from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Reshape, Dropout, Conv2D, LSTM, Layer, Flatten
from tensorflow.keras.optimizers import Adam

from models import split_data_and_labels, JensModelBuilder
from models.JensModel import jens_convert
from models.ModelBuilder import ModelBuilder
from sonar_types import Window


def create_deep_conv_lstm(config: DeepConvBuilder, return_layers: bool = False):
    """
    Based on https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/
    DeepConvModel
    """
    input_layer = Input(shape=(config.n_timesteps, config.n_features, 1), name="sensor_input")
    x = Dropout(.4)(input_layer)
    for n_filter in config.n_filters:
        x = Conv2D(filters=n_filter, kernel_size=(config.kernel_size, 1), strides=(config.stride_size, 1),
                   activation='relu')(x)

    x = Reshape((x.shape[1], -1))(x)
    x = LSTM(config.n_lstm_layers, return_sequences=True)(x)
    x = LSTM(config.n_lstm_layers, return_sequences=config.return_last_as_sequence)(x)
    if config.return_last_as_sequence:
        x = Flatten()(x)
    last_lstm_output = x
    x = Dense(config.n_outputs, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    if not return_layers:
        return model
    else:
        return model, input_layer, last_lstm_output


@dataclass
class DeepConvBuilder(ModelBuilder):
    n_outputs: int
    n_features: int
    n_timesteps: int

    model_name: str = "DeepConvLSTM"
    model_builder = create_deep_conv_lstm
    convert = jens_convert

    kernel_size: int = 15
    stride_size: int = 1
    n_lstm_layers: int = 32
    n_filters: list[int] = field(default_factory=lambda: [64, 64, 64])
    return_last_as_sequence: bool = False

    # 1/2 the default
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate = 0.00005

    def build_with_layers(self) -> tuple[Model, Layer, Layer]:
        model, i, o = self.model_builder(return_layers=True)
        return model, i, o


def deepconv_convert(config: DeepConvBuilder, windows: list[Window]) -> tuple[np.array, np.array]:
    X, y = split_data_and_labels(config, windows)
    return X, y
