from utils.cache_recordings import load_recordings
from utils.typing import assert_type
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_sonar_dataset import load_sonar_dataset
import itertools
import json
import os

import pandas as pd
from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Config Interface:
    (can be used generic)
        data_config.activity_idx_to_activity_name(activity_idx) (subclasses need to define the mapping that is required for that)
        data_config.load_dataset()
    """

    # Dataset Config (subclass responsibility) -----------

    raw_label_to_activity_idx_map = None
    raw_subject_to_subject_idx_map = None

    activity_idx_to_activity_name_map = None
    subject_idx_to_subject_name_map = None

    timestep_frequency = None # Hz

    timestep_frequency = None # Hz

    # interface (subclass responsibility to define) ------------------------------------------------------------

    def load_dataset(self) -> "list[Recording]":
        raise NotImplementedError(
            "init subclass of Config that defines the method activity_idx_to_activity_name"
        )

    # generic
    def raw_label_to_activity_idx(self, label: str) -> int:
        """
        from the label as it is saved in the dataset, to the activity index
        (Relabeling)
        """
        assert (
            self.raw_label_to_activity_idx_map is not None
        ), "A subclass of Config which initializes the var raw_label_to_activity_idx_map should be used to access activity mapping."
        return self.raw_label_to_activity_idx_map[label]

    def raw_subject_to_subject_idx(self, subject: str) -> int:
        assert (
            self.raw_subject_to_subject_idx_map is not None
        ), "A subclass of Config which initializes the var raw_subject_to_subject_idx_map should be used to access subject mapping."
        return self.raw_subject_to_subject_idx_map[subject]

    def subject_idx_to_subject_name(self, subject_idx: int) -> str:
        assert (
            self.subject_idx_to_subject_name_map is not None
        ), "A subclass of Config which initializes the var subject_idx_to_subject_name_map should be used to access subject mapping."
        assert_type((subject_idx, int))
        return self.subject_idx_to_subject_name_map[subject_idx]

    def activity_idx_to_activity_name(self, activity_idx: int) -> str:
        assert (
            self.activity_idx_to_activity_name_map is not None
        ), "A subclass of Config which initializes the var activity_idx_to_activity_name_map should be used to access activity mapping."
        assert_type((activity_idx, int))
        return self.activity_idx_to_activity_name_map[activity_idx]

    def n_activities(self) -> int:
        assert (
            self.activity_idx_to_activity_name_map is not None
        ), "A subclass of Config which initializes the var activity_idx_to_activity_name_map should be used to access activity mapping."
        return len(self.activity_idx_to_activity_name_map)


class OpportunityConfig(DataConfig):

    timestep_frequency = 30 # Hz
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

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

    def load_dataset(self) -> "list[Recording]":
        return load_opportunity_dataset(self.dataset_path)


class SonarConfig(DataConfig):

    timestep_frequency = 60 # Hz

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        labels = list(
            itertools.chain.from_iterable(
                category["entries"] for category in self.category_labels
            )
        )
        self.raw_label_to_activity_idx_map = {
            label: i for i, label in enumerate(labels)
        }  # no relabeling applied
        activities = {k: v for v, k in enumerate(labels)}
        self.activity_idx_to_activity_name_map = {v: k for k, v in activities.items()}

        self.raw_subject_to_subject_idx_map = {
            key: value for value, key in enumerate(self.raw_subject_label)
        }
        self.subject_idx_to_subject_name_map = {
            v: k for k, v in self.raw_subject_to_subject_idx_map.items()
        }  # just the inverse, do relabeling here, if needed

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def load_dataset(self, **args) -> "list[Recording]":
        return load_sonar_dataset(self.dataset_path, **args)

    raw_subject_label = [
        "unknown",
        "christine",
        "aileen",
        "connie",
        "yvan",
        "brueggemann",
        "jenny",
        "mathias",
        "kathi",
        "anja",
    ]
    category_labels = [
        {
            "category": "Others",
            "entries": [
                "invalid",
                "null - activity",
                "aufräumen",
                "aufwischen (staub)",
                "blumen gießen",
                "corona test",
                "kaffee kochen",
                "schrank aufräumen",
                "wagen schieben",
                "wäsche umräumen",
                "wäsche zusammenlegen",
            ],
        },
        {
            "category": "Morgenpflege",
            "entries": [
                "accessoires (parfüm) anlegen",
                "bad vorbereiten",
                "bett machen",
                "bett beziehen",
                "haare kämmen",
                "hautpflege",
                "ikp-versorgung",
                "kateterleerung",
                "kateterwechsel",
                "medikamente geben",
                "mundpflege",
                "nägel schneiden",
                "rasieren",
                "umkleiden",
                "verband anlegen",
            ],
        },
        {
            "category": "Waschen",
            "entries": [
                "duschen",
                "föhnen",
                "gegenstand waschen",
                "gesamtwaschen im bett",
                "haare waschen",
                "rücken waschen",
                "waschen am waschbecken",
                "wasser holen",
            ],
        },
        {
            "category": "Mahlzeiten",
            "entries": [
                "essen auf teller geben",
                "essen austragen",
                "essen reichen",
                "geschirr austeilen",
                "geschirr einsammeln",
                "getränke ausschenken",
                "getränk geben",
                "küche aufräumen",
                "küchenvorbereitungen",
                "tablett tragen",
            ],
        },
        {
            "category": "Assistieren",
            "entries": [
                "arm halten",
                "assistieren - aufstehen",
                "assistieren - hinsetzen",
                "assistieren - laufen",
                "insulingabe",
                "patient umlagern (lagerung)",
                "pflastern",
                "rollstuhl modifizieren",
                "rollstuhl schieben",
                "rollstuhl transfer",
                "toilettengang",
            ],
        },
        {
            "category": "Organisation",
            "entries": [
                "arbeiten am computer",
                "dokumentation",
                "medikamente stellen",
                "telefonieren",
            ],
        },
    ]

class Sonar22CategoriesConfig(DataConfig):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        self.raw_label_to_activity_idx_map = self.category_labels # no relabeling applied
        self.activity_idx_to_activity_name_map = {k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def load_dataset(self, **args) -> "list[Recording]":
        return load_recordings(self.dataset_path,self.raw_label_to_activity_idx_map, **args)

    category_labels = {'rollstuhl transfer': 0, 'essen reichen': 1, 'umkleiden': 2, 'bad vorbereiten': 3, 'bett machen': 4, 'gesamtwaschen im bett': 5, 'aufräumen': 6, 'geschirr einsammeln': 7, 'essen austragen': 8, 'getränke ausschenken': 9, 'küchenvorbereitung': 10, 'waschen am waschbecken': 11, 'rollstuhl schieben': 12, 'mundpflege': 13, 'haare kämmen': 14, 'essen auf teller geben': 15, 'dokumentation': 16, 'aufwischen (staub)': 17, 'haare waschen': 18, 'medikamente stellen': 19, 'accessoires anlegen': 20, 'föhnen': 21}

class LabPoseConfig(DataConfig):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        self.raw_label_to_activity_idx_map = self.category_labels # no relabeling applied
        self.activity_idx_to_activity_name_map = {k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def load_dataset(self, **args) -> "list[Recording]":
        return load_recordings(self.dataset_path,self.raw_label_to_activity_idx_map, **args)

    people = [
        "unknown",
        "marco",
        "felix",
        "valentin",
        "orhan",
        "kirill",
        "daniel",
        "alex",
        "lucas",
        "tobi",
        "franz"
    ]

    category_labels = {
        "medikamente stellen": 0,
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
        "rollstuhl schieben": 11,
        "rollstuhl transfer": 12
    }
    #category_labels = {'rollstuhl transfer': 0, 'essen reichen': 1, 'umkleiden': 2, 'bad vorbereiten': 3, 'bett machen': 4, 'gesamtwaschen im bett': 5, 'aufräumen': 6, 'geschirr einsammeln': 7, 'essen austragen': 8, 'getränke ausschenken': 9, 'küchenvorbereitung': 10, 'waschen am waschbecken': 11, 'rollstuhl schieben': 12, 'mundpflege': 13, 'haare kämmen': 14, 'essen auf teller geben': 15, 'dokumentation': 16, 'aufwischen (staub)': 17, 'haare waschen': 18, 'medikamente stellen': 19, 'accessoires anlegen': 20, 'föhnen': 21, 'null - activity': 22, 'ikp versorgung': 23}
