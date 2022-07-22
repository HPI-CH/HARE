import itertools
import json
import os


def init(data_config=None):
    """
    Note:
    - no typing of data_config possible (circular import)

    Refactoring idea:
    - pass the mapping, that we can easily switch between datasets and labels
    - mapping.py file (in utils) should include activity and subject mappings for the datasets
    - the experiments loads the required ones and passes them in the init (settings.init(mappings)O
    """

    global DATA_CONFIG
    DATA_CONFIG = data_config

    # System Config ---------------------------------------

    global IS_WINDOWS
    IS_WINDOWS = os.name == "nt"

    # Paths -----------------------------------------------

    global SAVED_EXPERIMENTS_PATH
    SAVED_EXPERIMENTS_PATH = "src/saved_experiments"

    global BP_PATH
    BP_PATH = "/dhc/groups/bp2021ba1"

    global ML_RAINBOW_PATH
    ML_RAINBOW_PATH = BP_PATH + "/apps/ml-rainbow"

    global DATA_PATH
    DATA_PATH = (
        BP_PATH + "/data"
        if not IS_WINDOWS
        else os.path.dirname(os.path.abspath(__file__)) + "/../dataWindows"
    )
