import numpy as np
import tensorflow as tf
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


class DeepConvLSTM(RainbowModel):
    """
    Based on https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition
    """

    def _create_model(self):
        input_layer = Input(
            shape=(self.window_size, self.n_features), name="sensor_input"
        )
        x = input_layer
        # x = self._preprocessing_layer(x)
        x = Reshape((self.window_size, self.n_features, 1))(x)
        x = Dropout(0.4)(x)
        for n_filter in [64, 64, 64, 64]:
            x = Conv2D(
                filters=n_filter, kernel_size=(5, 1), strides=(5, 1), activation="relu"
            )(x)

        x = Reshape((x.shape[1], -1))(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(128, return_sequences=False)(x)
        x = Dense(self.n_outputs, activation="softmax")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=self.metrics,
        )
        model.summary()
        return model
