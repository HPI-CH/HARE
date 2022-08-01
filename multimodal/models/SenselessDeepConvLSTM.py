import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.RainbowModel import RainbowModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv1D,
    Dense,
    Flatten,
    Dropout,
    LSTM,
    GlobalMaxPooling1D,
    MaxPooling2D,
    BatchNormalization,
    concatenate,
    Reshape,
    Permute,
    LSTM,
)


class SenselessDeepConvLSTM(RainbowModel):
    """
    https://github.com/STRCWearlab/DeepConvLSTM
    
    """

    def _create_model(self):

        # Config
        initializer = Orthogonal()

        def assert_data_dim(n_dim):
            def assertion(the_input):
                assert (
                    len(the_input.shape) == n_dim + 1
                ), f"Got shape {the_input.shape} with data dim {len(the_input.shape) - 1} but expected {n_dim}"

            return assertion

        def conv_layer(n_filters):
            def conv(the_input):
                assert_data_dim(n_dim=3)(the_input)
                # return Conv2D(filters=n_filters, kernel_size=(5, 1), strides=(5, 1), activation="relu")(the_input)
                return Conv1D(
                    filters=n_filters, kernel_size=5, strides=5, activation="relu"
                )(the_input)

            return conv

        def lstm_layer(units, return_sequence=True):
            def lstm(the_input):
                expected_dims = 2 if return_sequence else 1
                assert_data_dim(n_dim=expected_dims)(the_input)
                return LSTM(units=units, dropout=0.0, return_sequences=return_sequence)(
                    the_input
                )

            return lstm

        def reshape_3d_to_2d(the_input):
            assert_data_dim(n_dim=3)(the_input)
            return Reshape(
                target_shape=(
                    the_input.shape[1],
                    the_input.shape[2] * the_input.shape[3],
                )
            )(the_input)

        def reshape_2d_add_1_dim(the_input):
            assert_data_dim(n_dim=2)(the_input)
            return Reshape(target_shape=(the_input.shape[1], the_input.shape[2], 1))(
                the_input
            )

        # Model
        i = Input(shape=(self.window_size, self.n_features))

        x = lstm_layer(units=32)(i)

        x = reshape_2d_add_1_dim(x)
        x = conv_layer(n_filters=64)(x)

        x = reshape_3d_to_2d(x)
        x = lstm_layer(units=32)(x)

        x = Dense(units=400, activation="relu")(x)
        x = Dropout(0.5)(x)

        x = reshape_2d_add_1_dim(x)
        x = conv_layer(n_filters=64)(x)

        x = Flatten()(x)
        x = Dense(units=self.n_outputs, activation="softmax")(x)

        model = Model(i, x)
        model.compile(
            optimizer="Adam",
            loss="CategoricalCrossentropy",  # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
            metrics=["accuracy"],
        )

        return model
