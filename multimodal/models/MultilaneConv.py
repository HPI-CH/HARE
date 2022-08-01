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
)


class MultilaneConv(JensModel):
    """
    - leave recording out
    - higher level opportunity
    - 3 sec window (90)
    -> accuracy: 0.65
    """

    def _create_model(self):
        def conv_pooling_dropout(n_filters, kernel_size=(3, 3)):
            def layer_block(x):
                x = Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu")(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.3)(x)
                return x

            return layer_block

        i = Input(shape=(self.window_size, self.n_features, 1))

        outputs = []
        n_parallel_lines = 1
        for _ in range(n_parallel_lines):
            x = conv_pooling_dropout(32)(i)
            x = conv_pooling_dropout(64)(x)
            x = Flatten()(x)
            x = Dropout(0.2)(x)
            outputs.append(x)

        y = concatenate(outputs)
        y = Dense(1024, activation="relu")(y)
        y = Dropout(0.2)(y)
        y = Dense(self.n_outputs, activation="softmax")(y)

        model = Model(i, y)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="CategoricalCrossentropy",
            metrics=["accuracy"],
        )

        return model
