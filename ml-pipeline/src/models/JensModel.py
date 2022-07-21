from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.base_layer import Layer

from models import ModelBuilder, split_data_and_labels
from sonar_types import Window


def create_jens_model(config: JensModelBuilder, return_layers: bool = False):
    print(f"Building model: ({config.n_timesteps},{config.n_features}) input, {config.n_outputs} outputs")

    i = Input(shape=(config.n_timesteps, config.n_features, 1))
    x = Conv2D(32, (3, 3), strides=2, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.0005))(i)
    x = BatchNormalization()(x)
    # x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), strides=2, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), strides=2, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    # x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    last_layer_before_output = x
    x = Dropout(0.2)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(config.n_outputs, activation="softmax")(x)
    model = Model(i, x)
    jens_compile(config, model)
    if return_layers:
        return model, i, last_layer_before_output
    else:
        return model


def jens_compile(config: JensModelBuilder, model: Model):
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def jens_convert(config: JensModelBuilder, windows: list[Window]) -> tuple[np.array, np.array]:
    X, y = split_data_and_labels(config, windows)
    return np.expand_dims(X, -1), y


@dataclass
class JensModelBuilder(ModelBuilder):
    model_name = "Jens Model"
    model_builder = create_jens_model
    convert = jens_convert

    n_outputs: int
    n_features: int
    n_timesteps: int

    # Not needed for model building, but for easy passing
    n_epochs: int = 10
    batch_size: int = 64
    learning_rate = 0.00005

    def build_with_layers(self) -> tuple[Model, Layer, Layer]:
        model, i, o = self.model_builder(return_layers=True)
        return model, i, o
