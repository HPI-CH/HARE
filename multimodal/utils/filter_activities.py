from typing import Callable

import numpy as np
import pandas as pd

from utils.Recording import Recording


def filter_activities(
    recordings: "list[Recording]", activities_to_keep: list
) -> "list[Recording]":
    return filter_activities_custom(
        recordings, lambda activities: activities.isin(activities_to_keep)
    )


def filter_activities_negative(
    recordings: "list[Recording]", activities_to_remove: list
) -> "list[Recording]":
    return filter_activities_custom(
        recordings, lambda activities: ~activities.isin(activities_to_remove)
    )


def filter_activities_custom(
    recordings: "list[Recording]", filter_fn: Callable[[pd.Series], list[bool]]
) -> "list[Recording]":
    """
    Removes all activities where filter_fn is false
    """

    for recording in recordings:
        recording.activities.reset_index(drop=True, inplace=True)
        recording.sensor_frame.reset_index(drop=True, inplace=True)
        recording.time_frame.reset_index(drop=True, inplace=True)

        recording.activities = recording.activities[filter_fn(recording.activities)]
        recording.sensor_frame = recording.sensor_frame.loc[recording.activities.index]
        recording.time_frame = recording.time_frame.loc[recording.activities.index]

    return recordings


def filter_short_activities(
    recordings: "list[Recording]", threshhold: int = 3, strategy: int = 0
) -> "list[Recording]":
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
                        indices[i] : indices[i + 1]
                    ] = recording.activities.iloc[indices[i - 1]]
                elif strategy == 1:
                    recording.activities.iloc[
                        indices[i] : indices[i + 1]
                    ] = "null-activity"

    return recordings


def rename_activities(
    recordings: "list[Recording]", rules: dict = {}
) -> "list[Recording]":
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
