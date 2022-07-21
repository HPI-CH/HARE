from loading import reorder_sensor_columns
from sonar_types import PreprocessingFunction, Recording


def remove_columns(columns_to_remove: list[str]) -> PreprocessingFunction:
    """
    Removes all columns mentioned, if they exist
    """

    def fn(recordings: list[Recording]):
        for recording in recordings:
            recording.sensor_frame.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        return recordings

    return fn


def remove_columns_all_sensors(columns_to_remove: list[str], sensor_suffices: list[str]):
    """
    Removes all columns given by combinations of columns_to_remove and sensor_suffices
    """
    all_columns_to_remove = [f"{col}_{sfx}" for col in columns_to_remove for sfx in sensor_suffices]

    return remove_columns(all_columns_to_remove)


def keep_columns_all_sensors(columns_to_keep: list[str], sensor_suffices: list[str]):
    """
    Keeps all columns given.

    Only works if the columns are constant over recordings (first one is taken as representative)
    """
    all_columns_to_keep = [f"{col}_{sfx}" for col in columns_to_keep for sfx in sensor_suffices]

    return keep_columns(all_columns_to_keep)


def keep_columns(columns_to_keep: list[str]):
    """
    Keeps all columns given.

    Only works if the columns are constant over recordings (first one is taken as representative)
    """
    def fn(recordings: list[Recording]):
        columns_to_remove = recordings[0].sensor_frame.columns.drop(columns_to_keep)
        for recording in recordings:
            recording.sensor_frame.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        return recordings

    return fn


def reorder_columns(*, order) -> PreprocessingFunction:
    def reorder(recs: list[Recording]):
        for rec in recs:
            rec.sensor_frame = reorder_sensor_columns('', rec.sensor_frame, order)

        return recs
    return reorder
