from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Dropout

from models import JensModelBuilder, jens_compile


def build_multiwindow_model(model_config: MultiWindowConfig, base_input: Layer, base_output: Layer):
    base_model = Model(base_input, base_output)
    base_model.trainable = False

    parallel_outputs = []
    all_inputs = Input((model_config.n_windows_multi_window, model_config.n_timesteps, model_config.n_features, 1))
    for i in tf.unstack(all_inputs, axis=1):
        single_window_output = base_model(i)
        parallel_outputs += [single_window_output]

    windows_output = Concatenate()(parallel_outputs)
    x = Dense(1024, activation="relu")(windows_output)
    x = Dropout(0.2)(x)
    output = Dense(model_config.n_outputs, activation="softmax")(x)
    model = Model(inputs=all_inputs, outputs=output)
    jens_compile(model_config, model)
    model.summary()
    return model


@dataclass
class MultiWindowConfig(JensModelBuilder):
    model_name = "Multi Window Model"
    model_builder = build_multiwindow_model

    # n_outputs: int
    # n_features: int
    # n_timesteps: int
    n_windows_multi_window: int = 5

    # Not needed for model building, but for easy passing
    n_epochs: int = 10
    batch_size: int = 64

    def build(self, base_input: Layer, base_output: Layer) -> Model:
        return self.model_builder(base_input, base_output)

    def build_with_layers(self) -> tuple[Model, Layer, Layer]:
        raise NotImplementedError()
