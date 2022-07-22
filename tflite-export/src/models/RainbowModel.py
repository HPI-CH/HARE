# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

import os
from abc import abstractmethod
from typing import Any
from typing import Union
import numpy as np

# pylint: disable=g-direct-tensorflow-import
import tensorflow as tf
import inspect
import wandb
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore
from tflite_support import metadata as _metadata
from models.metadata_populator import MetadataPopulatorForTimeSeriesClassifier
from utils.folder_operations import create_folders_in_path
from utils.typing import assert_type
from tensorflow.keras.layers import (Dense)


class TimeSeriesModelSpecificInfo(object):
    """Holds information that is specifically tied to a time series classifier"""

    def __init__(
        self,
        name,
        version,
        window_size,
        sampling_rate,
        features,
        device_tags,
        class_labels,
        author,
        description,
    ):
        self.name = name
        self.version = version
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.features = features
        self.device_tags = device_tags
        self.class_labels = class_labels
        self.author = author
        self.description = description


class RainbowModel(tf.Module):
    # general
    model_name = "model"
    model: Any = None

    # Input Params
    n_outputs: Union[int, None] = None
    window_size: Union[int, None] = None
    stride_size: Union[int, None] = None
    class_weight = None

    model: Union[tf.keras.Model, None] = None
    batch_size: Union[int, None] = None
    verbose: Union[int, None] = None
    n_epochs: Union[int, None] = None
    n_features: Union[int, None] = None
    n_outputs: Union[int, None] = None
    learning_rate: Union[float, None] = None

    # Config
    wandb_project: Union[str, None] = None
    verbose: Union[int, None] = 1
    kwargs = None
    callbacks = []

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Child classes should build a model, assign it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.

        Base parameters:
        - `input_distribution_mean` - array of float values: mean of distribution of each feature
        - `input_distribution_variance` - array of float values: variance of distribution of each feature
        --> These parameters will be used by the normalization layer (accessible by the function _preprocessing_layer
            in the child classes _create_model method)
        """

        # per feature measures of input distribution
        self.input_distribution_mean = kwargs.get(
            "input_distribution_mean", None)
        self.input_distribution_variance = kwargs.get(
            "input_distribution_variance", None
        )

        # input size
        self.window_size = kwargs.get("window_size", None)
        self.n_features = kwargs.get("n_features", None)
        self.stride_size = kwargs.get("stride_size", self.window_size)

        # output size
        self.n_outputs = kwargs.get("n_outputs", None)

        # training
        self.batch_size = kwargs.get("batch_size", None)
        self.n_epochs = kwargs.get("n_epochs", None)
        self.learning_rate = kwargs.get("learning_rate", None)
        self.validation_split = kwargs.get("validation_split", 0.2)
        self.class_weight = kwargs.get("class_weight", None)
        self.metrics = kwargs.get("metrics", ["accuracy"])

        # others
        self.author = kwargs.get("author", "Unknown")
        self.version = kwargs.get("version", "Unknown")
        self.description = self._create_description()
        self.wandb_config = kwargs.get("wandb_config", None)
        self.verbose = kwargs.get("verbose", 1)
        self.model_name = kwargs.get("name", "model")
        self.kwargs = kwargs

        # Important declarations
        assert self.window_size is not None, "window_size is not set"
        assert self.n_features is not None, "n_features is not set"
        assert self.n_outputs is not None, "n_outputs is not set"
        assert self.batch_size is not None, "batch_size is not set"
        assert self.n_epochs is not None, "n_epochs is not set"
        assert self.learning_rate is not None, "learning_rate is not set"

        self.model = self._create_model()
        self.model.summary()

    def get_params(self) -> dict:
        """
        gets the parameters of the model
        """
        return dict(
            window_size=self.window_size,
            n_features=self.n_features,
            n_outputs=self.n_outputs,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            validation_split=self.validation_split,
            class_weight=self.class_weight,
            metrics=self.metrics,
            author=self.author,
            version=self.version,
            description=self.description,
            wandb_config=self.wandb_config,
            verbose=self.verbose,
            kwargs=self.kwargs,
        )

    def _create_model(self) -> tf.keras.Model:
        """
        Subclass Responsibility:
        returns a keras model
        """
        raise NotImplementedError

    def _create_description(self) -> Union[str, None]:
        """
        Subclass Responsibility:
        returns a string describing the model
        """
        return None

    def _preprocessing_layer(
        self, input_layer: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        x = tf.keras.layers.Normalization(
            axis=-1,
            variance=self.input_distribution_variance,
            mean=self.input_distribution_mean,
        )(input_layer)
        return x

    # The 'train' function takes input windows and a labels

    # Fit ----------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the self.model to the data
        """
        assert_type(
            [(X_train, (np.ndarray, np.generic)),
             (y_train, (np.ndarray, np.generic))]
        )
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train have to have the same length"

        # Wandb
        callbacks = None
        if self.wandb_config is not None:
            assert (
                self.wandb_config["project"] is not None
            ), "Wandb project name is not set"
            assert (
                self.wandb_config["entity"] is not None
            ), "Wandb entity name is not set"
            assert self.wandb_config["name"] is not None, "Wandb name is not set"

            wandb.init(
                project=str(self.wandb_config["project"]),
                entity=self.wandb_config["entity"],
                name=str(self.wandb_config["name"]),
                settings=wandb.Settings(start_method="fork"),
            )
            wandb.config = {
                "learning_rate": self.learning_rate,
                "epochs": self.n_epochs,
                "batch_size": self.batch_size,
            }
            callbacks = [wandb.keras.WandbCallback()]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_split=self.validation_split,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            callbacks=callbacks,
        )

    # Predict ------------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        return self.model.evaluate(X_test, y_test)

    def export(
        self,
        path: str,
        device_tags: "list[str]",
        features: "list[str]",
        class_labels: "list[str]",
    ) -> None:
        """
        will create an 'export' folder in the path, and save the model there in 3 different formats
        """
        print("Exporting model ...")

        # Define, create folder structure
        export_path = os.path.join(path, "export")
        export_path_raw_model = os.path.join(export_path, "raw_model")
        create_folders_in_path(export_path_raw_model)

        # Function signatures to export
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, self.window_size,
                                     self.n_features], dtype=tf.float32),
            ]
        )
        def infer(input_window):
            logits = self.model(input_window)
            probabilities = tf.nn.softmax(logits, axis=-1)
            return {"output": probabilities}

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, self.window_size,
                                     self.n_features], dtype=tf.float32),
                # Binary Classification
                tf.TensorSpec(shape=[None, self.n_outputs], dtype=tf.float32),
            ]
        )
        def train(input_windows, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(input_windows)
                print(self.model)
                loss = self.model.loss(labels, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )
            result = {"loss": loss}
            return result

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def restore(checkpoint_path):
            restored_tensors = {}
            for var in self.model.weights:
                restored = tf.raw_ops.Restore(
                    file_pattern=checkpoint_path,
                    tensor_name=var.name,
                    dt=var.dtype,
                    name="restore",
                )
                var.assign(restored)
                restored_tensors[var.name] = restored
            return restored_tensors

        @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
        def save(checkpoint_path):
            tensor_names = [weight.name for weight in self.model.weights]
            tensors_to_save = [weight.read_value()
                               for weight in self.model.weights]
            tf.raw_ops.Save(
                filename=checkpoint_path,
                tensor_names=tensor_names,
                data=tensors_to_save,
                name="save",
            )
            return {
                "checkpoint_path": checkpoint_path,
            }

        # 1/2 Export raw model ------------------------------------------------------------
        tf.saved_model.save(
            self.model,
            export_path_raw_model,
            signatures={
                "train": train.get_concrete_function(),
                "infer": infer.get_concrete_function(),
                "save": save.get_concrete_function(),
                "restore": restore.get_concrete_function(),
            },
        )
        # 2/2 Convert raw model to tflite model -------------------------------------------
        converter = tf.lite.TFLiteConverter.from_saved_model(
            export_path_raw_model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        with open(os.path.join(export_path, f"{self.model_name}.tflite"), "wb") as f:
            f.write(tflite_model)

        print("Export finished")

        self.populate_and_export_metadata(
            export_path, device_tags, features, class_labels
        )

    def _build_model_info(
        self, device_tags: "list[str]", features: "list[str]", class_labels: "list[str]"
    ):
        return TimeSeriesModelSpecificInfo(
            name=self.model_name,
            version=self.version,
            window_size=self.window_size,
            sampling_rate=60,  # TODO: get this from the data timestamps
            features=features,
            device_tags=device_tags,
            class_labels=class_labels,
            author=self.author,
            description=self.description,
        )

    def _write_associated_files(
        self,
        export_path: str,
        device_tags: "list[str]",
        features: "list[str]",
        class_labels: "list[str]",
    ):
        """
        writes the label file, the features file and the device_tags file
        """
        print("Writing associated files ...")
        # Write label file
        label_file_name = "labels.txt"
        with open(os.path.join(export_path, label_file_name), "w") as f:
            for label in class_labels:
                f.write(f"{label}\n")

        # Write features file
        features_file_name = "features.txt"
        with open(os.path.join(export_path, features_file_name), "w") as f:
            for feature in features:
                f.write(f"{feature}\n")

        # Write device_tags file
        device_tags_file_name = "device_tags.txt"
        with open(os.path.join(export_path, device_tags_file_name), "w") as f:
            for device_tag in device_tags:
                f.write(f"{device_tag}\n")

        return label_file_name, features_file_name, device_tags_file_name

    def populate_and_export_metadata(
        self,
        export_path: str,
        device_tags: "list[str]",
        features: "list[str]",
        class_labels: "list[str]",
    ) -> None:
        label_file, feature_file, device_tags_file = self._write_associated_files(
            export_path, device_tags, features, class_labels
        )

        export_model_path = os.path.join(
            export_path, f"{self.model_name}.tflite")

        # Generate the metadata objects and put them in the model file
        populator = MetadataPopulatorForTimeSeriesClassifier(
            export_model_path,
            self._build_model_info(device_tags, features, class_labels),
            os.path.join(export_path, label_file),
            os.path.join(export_path, feature_file),
            os.path.join(export_path, device_tags_file),
        )
        populator.populate()

        # Validate the output model file by reading the metadata and produce
        # a json file with the metadata under the export path
        displayer = _metadata.MetadataDisplayer.with_model_file(
            export_model_path)
        export_json_file = os.path.join(export_path, self.model_name + ".json")
        json_file = displayer.get_metadata_json()
        with open(export_json_file, "w") as f:
            f.write(json_file)

        print("Finished populating metadata and associated file to the model:")
        print(export_model_path)
        print("The metadata json file has been saved to:")
        print(export_json_file)
        print("The associated file that has been been packed to the model is:")
        print(displayer.get_packed_associated_file_list())

    def save_weights(self, checkpoint_path: str):
        self.model.save_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str):
        self.model.load_weights(checkpoint_path)

    def freeze_non_dense_layers(self):
        # Set non dense layers to not trainable (freezing them)
        for layer in self.model.layers:
            layer.trainable = type(layer) == Dense
