from typing import Callable

import numpy as np
import pandas as pd

from sonar_types import Recording, RecordingProcessingFunction


def filter_empty() -> RecordingProcessingFunction:
    return lambda rs: list(filter(lambda r: len(r.sensor_frame) > 0, rs))


def filter_activities(activities_to_keep: list) -> RecordingProcessingFunction:
    return filter_activities_custom(
        lambda activities: activities.isin(activities_to_keep),
        new_idx=lambda activities: activities.replace({
            v: idx for idx, v in enumerate(activities_to_keep)
        })
    )


def filter_activities_negative(activities_to_remove: list) -> RecordingProcessingFunction:
    return filter_activities_custom(lambda activities: ~activities.isin(activities_to_remove))


def filter_activities_custom(
        filter_fn: Callable[[pd.Series], list[bool]],
        new_idx: Callable[[pd.Series], pd.Series] = lambda r: r,
) -> RecordingProcessingFunction:
    """
    Removes all activities where filter_fn is false
    """

    def fn(recordings: list[Recording]):
        for recording in recordings:
            recording.activities.reset_index(drop=True, inplace=True)
            recording.sensor_frame.reset_index(drop=True, inplace=True)
            recording.time_frame.reset_index(drop=True, inplace=True)

            recording.activities = new_idx(recording.activities[filter_fn(recording.activities)])
            recording.sensor_frame = recording.sensor_frame.loc[recording.activities.index]
            recording.time_frame = recording.time_frame.loc[recording.activities.index]

        return recordings

    return fn


def filter_short_activities(
        recordings: list[Recording], threshhold: int = 3, strategy: int = 0
) -> list[Recording]:
    """
    Replaces activities shorter than threshhold by [value]. [value] depends on strategy.
    strategy 0: replaces short activities with previous activity
    strategy 1: replaces short activities with 'null-activity'

    threshold: number of seconds
    """

    if strategy != 0 and strategy != 1:
        raise ValueError("strategy has to be 0 or 1")

    for recording in recordings:
        activities = np.array(recording.activities)
        indices = np.where(activities[:-1] != activities[1:])[0] + 1

        for i in range(len(indices) - 1):
            if recording.time_frame[indices[i + 1]] - recording.time_frame[
                indices[i]
            ] < (threshhold * 1000000):
                if strategy == 0:
                    recording.activities.iloc[
                    indices[i]: indices[i + 1]
                    ] = recording.activities.iloc[indices[i - 1]]
                elif strategy == 1:
                    recording.activities.iloc[
                    indices[i]: indices[i + 1]
                    ] = "null-activity"

    return recordings


def rename_activities(
        recordings: list[Recording], rules: dict = {}
) -> list[Recording]:
    """
    Renames / groups activities defined in rules.
    rules example structure:
    {
        'activity_name': 'new_activity_name',
    }
    """

    for recording in recordings:
        for old_activity, new_activity in rules.items():
            recording.activities[recording.activities == old_activity] = new_activity

    return recordings
