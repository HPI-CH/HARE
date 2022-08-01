import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models.RainbowModel import RainbowModel
from models.JensModel import JensModel
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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


class MultilaneConvLSTM(RainbowModel):
    """
    Best performer:
        - leave recording out
        - higher level opportunity
        - 3 sec window (90)
        -> accuracy: 0.65
    """

    def _create_model(self):

        i = Input(shape=(self.window_size, self.n_features))

        x = Reshape(target_shape=(self.window_size, self.n_features, 1))(i)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(rate=0.5)(x)

        y = LSTM(units=10, activation="relu")(i)
        y = Flatten()(y)

        out = concatenate(inputs=[x, y])
        out = Dense(units=self.n_outputs, activation="softmax")(out)

        model = Model(inputs=i, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="CategoricalCrossentropy",
            metrics=["accuracy"],
        )

        return model
