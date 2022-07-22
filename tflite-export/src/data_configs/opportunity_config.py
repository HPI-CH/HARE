from loader.load_opportunity_dataset import load_opportunity_dataset
from data_configs.data_config import DataConfig
from utils.Recording import Recording

class OpportunityConfig(DataConfig):

    timestep_frequency = 30  # Hz

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.original_idx_to_activity_idx_map = {
            0: 0,
            101: 1,
            102: 2,
            103: 3,
            104: 4,
            105: 5,
        }

        self.raw_label_to_activity_idx_map = {
            "null": 0,
            "relaxing": 1,
            "coffee time": 2,
            "early morning": 3,
            "cleanup": 4,
            "sandwich time": 5,
        }

        self.activity_idx_to_activity_name_map = {
            0: "null",
            1: "relaxing",
            2: "coffee time",
            3: "early morning",
            4: "cleanup",
            5: "sandwich time",
        }

    def _load_dataset(self,**kwargs) ->  "list[Recording]":
        return load_opportunity_dataset(self.dataset_path)