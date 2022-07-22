import itertools
import os

import numpy as np
import pandas as pd
from utils.data_set import DataSet

from utils.Recording import Recording
import utils.settings as settings


def load_opportunity_dataset(opportunity_dataset_path: str) -> "list[Recording]":
    """
    Returns a list of Recordings from the opportunity dataset
    """
    print("Will read the opportunity dataset")
    opportunity_dataset_path += "/dataset"
    subject_ids = range(1, 5)
    recording_ids = range(1, 6)

    # see loader/opportunity_col_names to make your selection
    selected_feature_names = [
        "IMU-BACK-accX",
        "IMU-BACK-accY",
        "IMU-BACK-accZ",
        "IMU-BACK-Quaternion1",
        "IMU-BACK-Quaternion2",
        "IMU-BACK-Quaternion3",
        "IMU-BACK-Quaternion4",
        "IMU-RLA-accX",
        "IMU-RLA-accY",
        "IMU-RLA-accZ",
        "IMU-RLA-Quaternion1",
        "IMU-RLA-Quaternion2",
        "IMU-RLA-Quaternion3",
        "IMU-RLA-Quaternion4",
        "IMU-LLA-accX",
        "IMU-LLA-accY",
        "IMU-LLA-accZ",
        "IMU-LLA-Quaternion1",
        "IMU-LLA-Quaternion2",
        "IMU-LLA-Quaternion3",
        "IMU-LLA-Quaternion4",
        "IMU-L-SHOE-EuX",
        "IMU-L-SHOE-EuY",
        "IMU-L-SHOE-EuZ",
        "IMU-L-SHOE-Nav_Ax",
        "IMU-L-SHOE-Nav_Ay",
        "IMU-L-SHOE-Nav_Az",
        "IMU-L-SHOE-Body_Ax",
        "IMU-L-SHOE-Body_Ay",
        "IMU-L-SHOE-Body_Az",
        "IMU-L-SHOE-AngVelBodyFrameX",
        "IMU-L-SHOE-AngVelBodyFrameY",
        "IMU-L-SHOE-AngVelBodyFrameZ",
        "IMU-L-SHOE-AngVelNavFrameX",
        "IMU-L-SHOE-AngVelNavFrameY",
        "IMU-L-SHOE-AngVelNavFrameZ",
        "IMU-R-SHOE-EuX",
        "IMU-R-SHOE-EuY",
        "IMU-R-SHOE-EuZ",
        "IMU-R-SHOE-Nav_Ax",
        "IMU-R-SHOE-Nav_Ay",
        "IMU-R-SHOE-Nav_Az",
        "IMU-R-SHOE-Body_Ax",
        "IMU-R-SHOE-Body_Ay",
        "IMU-R-SHOE-Body_Az",
        "IMU-R-SHOE-AngVelBodyFrameX",
        "IMU-R-SHOE-AngVelBodyFrameY",
        "IMU-R-SHOE-AngVelBodyFrameZ",
        "IMU-R-SHOE-AngVelNavFrameX",
        "IMU-R-SHOE-AngVelNavFrameY",
        "IMU-R-SHOE-AngVelNavFrameZ",
    ]
    print(f"Selected features (n_features: {len(selected_feature_names)}):\n", "\n".join(
        ["\t" + str(feature_name) for feature_name in selected_feature_names]))

    # Get column names
    col_names = []
    with open("src/loader/opportunity_col_names", "r") as file:
        lines = file.read().splitlines()
        for line in lines:
            col_names.append(line)

    recordings = []
    for sub, rec in itertools.product(subject_ids, recording_ids):
        file_name = f"S{sub}-ADL{rec}.dat"
        file_path = os.path.join(opportunity_dataset_path, file_name)
        print(f"Reading {file_path} ...")
        file_df = pd.read_csv(file_path, delimiter=" ", header=None)
        file_df.columns = col_names  # give them the real column names

        recordings.append(Recording(
            sensor_frame=file_df.loc[:, selected_feature_names],
            time_frame=file_df.loc[:, 'MILLISEC'],
            activities=file_df.loc[:, 'HL_Activity'].map(
                lambda label: settings.DATA_CONFIG.original_idx_to_activity_idx_map[label]

            ),  # Use `[0]` to get only one activity | maps 0, 101, 102, 103, 104, 105 to 0, 1, 2, 3, 4, 5
            subject=str(sub),
            recording_index=int(rec)
        ))

    print(f"\n => Total {len(recordings)} recordings read")

    return recordings
