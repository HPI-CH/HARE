from utils.cache_recordings import load_recordings
from data_configs.data_config import DataConfig
from utils.Recording import Recording


class SonarLab5Config(DataConfig):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.raw_label_to_activity_idx_map = self.category_labels  # no relabeling applied
        self.activity_idx_to_activity_name_map = {
            k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def _load_dataset(self, **kwargs) -> "list[Recording]":
        return load_recordings(self.dataset_path, self.raw_label_to_activity_idx_map, **kwargs)

    category_labels = {
        "aufräumen": 0,
        "bett machen": 1,
        "essen reichen": 2,
        "gesamtwaschen im bett": 3,
        "getränke ausschenken": 4,
        "rollstuhl schieben": 5,
    }
