import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.RainbowModel import RainbowModel
from models.JensModel import JensModel
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


class AlternativeDeepConvLSTM(RainbowModel):
    """
    https://github.com/STRCWearlab/DeepConvLSTM

    """

    def _create_model(self):

        initializer = Orthogonal()

        def conv_layer(n_filters): return lambda the_input: Conv2D(
            filters=n_filters,
            strides=(5, 1),
            kernel_size=(5, 1),
            activation="relu",
            kernel_initializer=initializer,
        )(the_input)

        def lstm_layer(the_input): return LSTM(
            units=32, dropout=0.0)(the_input)

        i = Input(shape=(self.window_size, self.n_features))
        x = self._preprocessing_layer(i)
        # Adding 4 CNN layers.
        x = Reshape(target_shape=(self.window_size, self.n_features, 1))(x)
        conv_n_filters = [32, 64]
        for n_filters in conv_n_filters:
            x = conv_layer(n_filters=n_filters)(x)

        x = Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3]),))(x)

        for _ in range(1):
            x = lstm_layer(x)

        x = Flatten()(x)
        x = Dense(units=self.n_outputs, activation="softmax")(x)

        model = Model(i, x)
        model.compile(
            optimizer="Adam",
            # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
            loss="CategoricalCrossentropy",
            metrics=["accuracy"],
        )

        return model
