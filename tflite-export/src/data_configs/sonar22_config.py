from utils.cache_recordings import load_recordings
from data_configs.data_config import DataConfig
from utils.Recording import Recording

class Sonar22CategoriesConfig(DataConfig):
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

    category_labels = {'rollstuhl transfer': 0, 'essen reichen': 1, 'umkleiden': 2, 'bad vorbereiten': 3, 'bett machen': 4, 'gesamtwaschen im bett': 5, 'aufräumen': 6, 'geschirr einsammeln': 7, 'essen austragen': 8, 'getränke ausschenken': 9, 'küchenvorbereitung': 10,
                       'waschen am waschbecken': 11, 'rollstuhl schieben': 12, 'mundpflege': 13, 'haare kämmen': 14, 'essen auf teller geben': 15, 'dokumentation': 16, 'aufwischen (staub)': 17, 'haare waschen': 18, 'medikamente stellen': 19, 'accessoires anlegen': 20, 'föhnen': 21}

    