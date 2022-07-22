from loader.load_gait_analysis_dataset import load_gait_analysis_dataset
from data_configs.data_config import DataConfig
from utils.Recording import Recording

class GaitAnalysisConfig(DataConfig):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.raw_label_to_activity_idx_map = self.category_labels
        self.activity_idx_to_activity_name_map = {
            k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        self.sensor_suffix_order = ["LF", "RF", "SA"]


    def _load_dataset(self, **kwargs) -> "list[Recording]":
        return load_gait_analysis_dataset(self.dataset_path, **kwargs)

    category_labels = {'fatigue': 0, 'non-fatigue': 1}
