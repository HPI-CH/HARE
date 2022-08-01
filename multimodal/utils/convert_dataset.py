from copy import deepcopy
from typing import List
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, RotationSpline

from utils import settings
from utils.Recording import Recording


def convert_quaternion_to_matrix(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting quaternion to matrix for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [
                f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y", "Z"]
            ]

            # The matrix is 3x3, so we need 5 more columns. We add these to the Quat-columns:
            added_rotation_cols = [
                f"Rot_{axis}_{sensor_suffix}" for axis in ["5", "6", "7", "8", "9"]
            ]
            # Insert the rows before writing to them is necessary
            for name in added_rotation_cols:
                recording.sensor_frame.insert(
                    recording.sensor_frame.columns.get_loc(quaternion_cols[-1]),
                    name,
                    np.nan,
                )
            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(
                recording.sensor_frame.loc[filled_rows, quaternion_cols]
            )

            # Convert to matrix and reshape them to be an array of 9 values
            matrices = quaternions.as_matrix().reshape((-1, 9))

            # Write the matrices to the quat + new columns
            recording.sensor_frame.loc[
                filled_rows, quaternion_cols + added_rotation_cols
            ] = matrices

            # Rename quaternion columns 
            recording.sensor_frame.rename(
                columns={
                    f"Quat_{axis}_{sensor_suffix}": f"Rot_{index+1}_{sensor_suffix}"
                    for (index, axis) in enumerate(["W", "X", "Y", "Z"])
                },
                inplace=True,
            )

    return recordings


def convert_quaternion_to_euler(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(
                f"Converting quaternion to euler angles for recording {idx}", end="\r"
            )

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [
                f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y", "Z"]
            ]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(
                recording.sensor_frame.loc[filled_rows, quaternion_cols]
            )

            # Convert to euler angles
            degrees = quaternions.as_euler("zyx")

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, quaternion_cols[:3]] = degrees
            recording.sensor_frame.drop(quaternion_cols[3], axis=1, inplace=True)

    return recordings


def convert_quaternion_to_vector(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting quaternion to euler angles for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            quaternion_cols = [
                f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y", "Z"]
            ]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[quaternion_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_quat(
                recording.sensor_frame.loc[filled_rows, quaternion_cols]
            )

            # Convert to euler angles
            vector = quaternions.apply([0, 0, 1])  # quaternions.as_rotvec()

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, quaternion_cols[:3]] = vector
            recording.sensor_frame.drop(quaternion_cols[3], axis=1, inplace=True)

    print()
    return recordings


def convert_euler_to_vector(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting euler to vectors for recording {idx}")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            euler_cols = [f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y"]]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[euler_cols[0]].notnull()

            # Read all quaternions simultaneously
            quaternions = R.from_euler(
                "zxy", recording.sensor_frame.loc[filled_rows, euler_cols]
            )

            # Convert to euler angles
            vector = quaternions.apply([0, 0, 1])  # quaternions.as_rotvec()

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, euler_cols] = vector

    print()
    return recordings


def convert_euler_to_velocity(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting to velocity for recording {idx}", end="\r")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            columns = [f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y"]]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[columns[0]].notnull()

            angles = recording.sensor_frame.loc[filled_rows, columns].values
            times = recording.time_frame.loc[filled_rows].values
            rotations = R.from_euler("zyx", angles)
            spline = RotationSpline(times, rotations)

            # Convert to euler angles
            velocity = spline(times, 1)

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, columns[:3]] = velocity

    print()
    return recordings


def convert_quat_to_velocity(recordings: List[Recording]) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting to velocity for recording {idx}", end="\r")

        # Iterate over all sensors - we need to convert this many quaternions
        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            # Build the column names, that we need to select for the quaternion
            columns = [f"Quat_{axis}_{sensor_suffix}" for axis in ["W", "X", "Y", "Z"]]

            # To not get an error when we try to read the quaternion, select only not nan rows
            filled_rows = recording.sensor_frame[columns[0]].notnull()

            quaternions = recording.sensor_frame.loc[filled_rows, columns].values
            times = recording.time_frame.loc[filled_rows].values
            rotations = R.from_quat(quaternions)
            spline = RotationSpline(times, rotations)

            # Convert to euler angles
            velocity = spline(times, 1)

            # Write them and remove the leftover column
            recording.sensor_frame.loc[filled_rows, columns[:3]] = velocity
            recording.sensor_frame.drop(columns[3], axis=1, inplace=True)

    print()
    return recordings


def convert_recording_speed(
    recordings: List[Recording], multiplier: float
) -> List[Recording]:
    recordings = deepcopy(recordings)

    # Iterate over all recordings
    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Changing speed for recording {idx}", end="\r")
        # Merge the time and sensor frames
        combined_frame = pd.merge(
            recording.time_frame,
            recording.sensor_frame,
            left_index=True,
            right_index=True,
        )
        combined_frame.resample("12ms").interpolate(method="spline")

    print()
    return recordings


def convert_to_relative_sensor_data(recordings: 'list[Recording]') -> 'list[Recording]':
    recordings = convert_quaternion_to_matrix(recordings)

    # Indices of columns of root sensor are the same for all recordings
    root_columns = [f"Rot_{num+1}_ST" for num in range(9)]
    root_columns_idx = [recordings[0].sensor_frame.columns.get_loc(col) for col in root_columns]

    root_acc_columns = [f"dv[{num+1}]_ST" for num in range(3)]
    root_acc_columns_idx = [recordings[0].sensor_frame.columns.get_loc(col) for col in root_acc_columns]

    for idx, recording in enumerate(recordings):
        if idx % 10 == 0:
            print(f"Converting to relative sensor data for recording {idx}", end="\r")

        # Calculate matrices and vectors for root sensor
        root_matrices = recording.sensor_frame.iloc[:, root_columns_idx].to_numpy()
        root_matrices = root_matrices.reshape(root_matrices.shape[0], 3, 3)
        inversed_root_matrices = np.linalg.inv(root_matrices)

        # Convert dv to acc
        recording.sensor_frame.iloc[:, root_acc_columns_idx] = recording.sensor_frame.iloc[:, root_acc_columns_idx] * 60
        root_acc_vectors = recording.sensor_frame.iloc[:, root_acc_columns_idx].to_numpy()

        for sensor_suffix in settings.DATA_CONFIG.sensor_suffix_order:
            if sensor_suffix == "ST":
                continue
            
            # Convert orientations
            ori_columns = [f"Rot_{num+1}_{sensor_suffix}" for num in range(9)]
            columns_idx = [recording.sensor_frame.columns.get_loc(col) for col in ori_columns]

            # We get a 2D array of rows
            ori_matrices = recording.sensor_frame.iloc[:, columns_idx].to_numpy()
            # Need to transform the rows into 3x3 matrices
            ori_matrices = ori_matrices.reshape(ori_matrices.shape[0], 3, 3)

            relative_matrices = np.matmul(inversed_root_matrices, ori_matrices)
            recording.sensor_frame.iloc[:, columns_idx] = relative_matrices.reshape(relative_matrices.shape[0], 9)

            # Convert accelerations
            acc_columns = [f"dv[{num+1}]_{sensor_suffix}" for num in range(3)]
            columns_idx = [recording.sensor_frame.columns.get_loc(col) for col in acc_columns]

            # Convert dv to acceleration
            recording.sensor_frame.iloc[:, columns_idx] = recording.sensor_frame.iloc[:, columns_idx] * 60

            acc_vectors = recording.sensor_frame.iloc[:, columns_idx].to_numpy()

            relative_vectors = acc_vectors - root_acc_vectors

            # Multiple all 3x3 matrices with the corresponding 3x1 vectors --> 3x1 vectors
            normalized_vectors = list()
            for i, matrix in enumerate(inversed_root_matrices):
                product = np.dot(matrix, relative_vectors[i])
                normalized_vectors.append(product)

            normalized_vectors = np.asarray(normalized_vectors)
            assert normalized_vectors.shape == relative_vectors.shape

            recording.sensor_frame.iloc[:, columns_idx] = normalized_vectors
    
    return recordings
