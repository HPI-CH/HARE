from tensorflow import keras
from tensorflow.keras import layers
from models.JensModel import JensModel


class MNISTModel(JensModel):
    """
    - leave recording out
    - higher level opportunity
    - 3 sec window (90)
    -> accuracy: 0.62
    """

    def _create_model(self):

        model = keras.Sequential(
            [
                keras.Input(shape=(self.window_size, self.n_features, 1)),
                self._preprocessing_layer,
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.n_outputs, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.summary()
        return model
