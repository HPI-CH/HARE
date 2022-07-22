from utils.cache_recordings import load_recordings
from data_configs.data_config import DataConfig
from utils.Recording import Recording


class SonarLabConfig(DataConfig):
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

    activity_idx_to_display_name_map = {
        0: "null - activity",
        1: "clean",
        2: "wipe dust",
        3: "make bed",
        4: "documentation",
        5: "dress",
        6: "serve food",
        7: "cleaning in bed",
        8: "pour drinks",
        9: "comb hair",
        10: "skin care",
        11: "prepare medicine",
        12: "push wheelchair",
        13: "wheelchair transfer",
    }

    raw_subject_to_subject_idx_map = {
        "orhan": 0,
        "daniel": 1,
        "felix": 2,
        "tobi": 3,
        "lucas": 4,
        "kirill": 5,
        "marco": 6,
        "valentin": 7,
        "alex": 8,
        "franz": 9,
    }
