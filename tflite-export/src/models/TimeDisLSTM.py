import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
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
    TimeDistributed,
)


class TimeDisLSTM(JensModel):
    """
    - leave recording out
    - higher level opportunity
    - 3 sec window (90)
    -> accuracy: 0.47
    """

    def _create_model(self):

        i = Input(shape=(self.window_size, self.n_features))
        x = self._preprocessing_layer(i)
        x = Reshape(target_shape=(self.window_size, self.n_features, 1))(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.2)(x)

        y = Permute((2, 1))(i)
        y = TimeDistributed(LSTM(8, activation="relu"))(y)
        y = Dropout(0.2)(y)
        y = Permute((2, 1, 3))(y)

        out = concatenate([x, y])
        out = Dense(self.n_outputs, activation="softmax")(out)

        model = Model(i, out)
        model.summary()
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="CategoricalCrossentropy",
            metrics=["accuracy"],
        )

        return model
