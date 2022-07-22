from tflite_support import metadata_schema_py_generated as _metadata_fb
import flatbuffers
from tflite_support import metadata as _metadata
import os


class MetadataPopulatorForTimeSeriesClassifier(object):
    """Populates the metadata for a time series classifier"""

    def __init__(
        self,
        model_file,
        model_info,
        label_file_path,
        sensor_data_file_path,
        sensor_file_path,
    ):
        self.model_info = model_info
        self.model_file = model_file
        self.label_file_path = label_file_path
        self.sensor_data_file_path = sensor_data_file_path
        self.sensor_file_path = sensor_file_path
        self.metadata_buf = None

    def populate(self):
        """Creates Metadata and the populates it for a time series classifier"""
        self._create_metadata()
        self._populate_metadata()

    def _create_metadata(self):
        """Creates the metadata for a time series classifier"""

        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_info.name
        model_meta.description = self.model_info.description
        model_meta.author = self.model_info.author
        model_meta.version = self.model_info.version
        model_meta.license = (
            "Apache License. Version 2.0 https://www.apache.org/licenses/LICENSE-2.0."
        )

        # Packs associated file for sensor data inputs.
        label_input_file = _metadata_fb.AssociatedFileT()
        label_input_file.name = os.path.basename(self.sensor_data_file_path)
        label_input_file.description = "Names of sensor data inputs."
        label_input_file.type = _metadata_fb.AssociatedFileType.DESCRIPTIONS

        # Packs associated file for sensors.
        sensor_file = _metadata_fb.AssociatedFileT()
        sensor_file.name = os.path.basename(self.sensor_file_path)
        sensor_file.description = "Names of sensors, that data was collected on."
        sensor_file.type = _metadata_fb.AssociatedFileType.DESCRIPTIONS
        model_meta.associatedFiles = [label_input_file, sensor_file]

        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "window"
        input_meta.description = (
            "Input window to be classified. The expected window has a size of {0} at a sampling "
            "rate of {1}. It gets data from the features: {2} and {3} IMU sensors at the "
            "positions: {4}".format(
                self.model_info.window_size,
                self.model_info.sampling_rate,
                self.model_info.features,
                len(self.model_info.device_tags),
                self.model_info.device_tags,
            )
        )
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
        input_meta.content.contentProperties.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties
        )
        # Creates output info.
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = (
            "Probabilities of the {0} labels respectively.".format(
                len(self.model_info.class_labels)
            )
        )
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats

        # Packs associated file for label outputs.
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for classification output."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        # Builds flatbuffer.
        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
        )
        self.metadata_buf = b.Output()

    def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        print("Populating metadata...")
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        print(
            f"Loading associated files... {self.label_file_path}, {self.sensor_file_path}, {self.sensor_data_file_path}"
        )
        populator.load_associated_files(
            [self.label_file_path, self.sensor_data_file_path, self.sensor_file_path]
        )
        populator.populate()

    def _normalization_params(self, feature_norm):
        """Creates normalization process unit for each input feature."""
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions
        )
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = feature_norm["mean"]
        input_normalization.options.std = feature_norm["std"]
        return input_normalization
