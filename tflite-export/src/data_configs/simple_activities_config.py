from utils.cache_recordings import load_recordings
from data_configs.data_config import DataConfig
from utils.Recording import Recording

class SimpleActivitiesConfig(DataConfig):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.raw_label_to_activity_idx_map = self.category_labels  # no relabeling applied
        self.activity_idx_to_activity_name_map = {
            k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def _load_dataset(self, **args) -> "list[Recording]":
        return load_recordings(self.dataset_path, self.raw_label_to_activity_idx_map, **args)

    category_labels = {
        "null - activity": 0,
        "aufräumen": 1,
        "aufwischen (staub)": 2,
        "bett machen": 3,
        "dokumentation": 4,
        "umkleiden": 5,
        "essen reichen": 6,
        "gesamtwaschen im bett": 7,
        "getränke ausschenken": 8,
        "haare kämmen": 9,
        "waschen am waschbecken": 10,
        "medikamente stellen": 11,
        "rollstuhl schieben": 12,
        "rollstuhl transfer": 13
    }