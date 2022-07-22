from loader.load_sonar_dataset import load_sonar_dataset
from data_configs.data_config import DataConfig
from utils.Recording import Recording

class SonarConfig(DataConfig):

    timestep_frequency = 60  # Hz

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        labels = list(
            itertools.chain.from_iterable(
                category["entries"] for category in self.category_labels
            )
        )
        self.raw_label_to_activity_idx_map = {
            label: i for i, label in enumerate(labels)
        }  # no relabeling applied
        activities = {k: v for v, k in enumerate(labels)}
        self.activity_idx_to_activity_name_map = {
            v: k for k, v in activities.items()}

        self.raw_subject_to_subject_idx_map = {
            key: value for value, key in enumerate(self.raw_subject_label)
        }
        self.subject_idx_to_subject_name_map = {
            v: k for k, v in self.raw_subject_to_subject_idx_map.items()
        }  # just the inverse, do relabeling here, if needed

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def _load_dataset(self, **args) -> "list[Recording]":
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
