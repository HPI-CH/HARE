from models.RainbowModel import RainbowModel
from models.JensModel import JensModel
from tensorflow.keras.layers import Conv1D, Dense, Dropout  # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape  # type: ignore
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout  # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy  # type: ignore
from tensorflow.keras.models import Model  # type: ignore


class OldLSTM(RainbowModel):

    """
    Our initial model

    - leave recording out
    - higher level opportunity
    - 3 sec window (90)
    -> accuracy: 0.47
    """

    def squeeze_excite_block(self, input):
        """ Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor
        Returns: a keras tensor
        """
        filters = input.shape[-1]  # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(
            filters // 16,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        se = Dense(
            filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        se = multiply([input, se])
        return se

    def _create_model(self):

        ip = Input(shape=(self.window_size, self.n_features))

        x = Permute((2, 1))(ip)
        x = LSTM(8)(x)
        x = Dropout(rate=0.4)(x)

        y = Conv1D(128, 8, padding="same", kernel_initializer="he_uniform")(ip)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(256, 5, padding="same", kernel_initializer="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = self.squeeze_excite_block(y)

        y = Conv1D(128, 3, padding="same", kernel_initializer="he_uniform")(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(self.n_outputs, activation="softmax")(x)

        model = Model(ip, out)
        model.compile(
            loss=CategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"]
        )

        return model
