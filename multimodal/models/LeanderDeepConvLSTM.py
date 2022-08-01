import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.RainbowModel import RainbowModel
from models.ResNetModel_Multimodal import ResNetModelMultimodal
from models.JensModel import JensModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import (
    Input,
    Conv2D,
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


class LeanderDeepConvLSTM(ResNetModelMultimodal):
    """
    https://github.com/STRCWearlab/DeepConvLSTM

    """

    def _create_model(self, n_features, n_outputs):

        initializer = Orthogonal()
        conv_layer = lambda n_filters: lambda the_input: Conv2D(
            filters=n_filters,
            strides=(5, 1),
            kernel_size=(5, 1),
            activation="relu",
            kernel_initializer=initializer,
        )(the_input)
        lstm_layer = lambda the_input: LSTM(
            units=32, dropout=0.1, return_sequences=True, kernel_initializer=initializer
        )(the_input)

        i = Input(shape=(self.window_size, self.n_features))

        # Adding 4 CNN layers.
        x = Reshape(target_shape=(self.window_size, self.n_features, 1))(i)
        conv_n_filters = [32, 64]
        for n_filters in conv_n_filters:
            x = conv_layer(n_filters=n_filters)(x)

        x = Reshape((int(x.shape[1]), int(x.shape[2]) * int(x.shape[3]),))(x)

        for _ in range(1):
            x = lstm_layer(x)

        x = Flatten()(x)
        x = Dense(units=n_outputs, activation="softmax")(x)

        model = Model(i, x)
        model.compile(
            optimizer="RMSprop",
            loss="CategoricalCrossentropy",  # CategoricalCrossentropy (than we have to to the one hot encoding - to_categorical), before: "sparse_categorical_crossentropy"
            metrics=["accuracy"],
        )

        return model
