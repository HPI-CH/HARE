from dataclasses import dataclass

import numpy as np

from utils.mappings import Mappings

np.set_printoptions(linewidth=140)


@dataclass
class Config:
    """
    Configuration class for the application.
    """
    # The path to the directory containing the data files.
    dataset_root_dir: str = "/dhc/groups/bp2021ba1/data"

    # The folder name of our dataset
    sonar_folder_name: str = 'SONAR'

    sonar_lab_data: str = 'lab_data'

    sonar_cached_name: str = 'combined_dataset'

    sonar_filtered_name: str = 'filtered_dataset'

    sonar_preprocessed_no_null: str = 'preprocessed_no_null_dataset'

    sonar_preprocessed_name: str = 'preprocessed_dataset'

    sonar_tfrecord_name: str = 'tfrecord/records100'

    log_level = 'verbose'  # | 'verbose' | 'error'

    # The folder of old sonar dataset
    sonar_old_folder_name: str = '5-sensor-all'

    # The order in which the sensor colums should be
    sonar_sensor_suffix_order = [
        "LF", "LW", "ST", "RW", "RF"
    ]

    sonar_column_names = [
        'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'dq_W', 'dq_X', 'dq_Y', 'dq_Z', 'dv[1]', 'dv[2]', 'dv[3]', 'Mag_X', 'Mag_Y', 'Mag_Z'
    ]

    # Same order as opportunity for transfer learning
    sonar_opp_sensor_order = [
        "ST", "RW", "LW", "LF", "RF"
    ]

    sonar_st1_sensor_map = Mappings.st1_sensor_map

    sonar_csv_header_size = 8

    sonar_people = Mappings.people

    sonar_labels = Mappings.labels

    # The folder name to OPPORTUNITY dataset
    opp_folder_name: str = "OPPORTUNITY"
