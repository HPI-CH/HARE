import itertools

from sonar_types import PreprocessingFunction, Recording


def map_activities_to_int(*, empty_activity_dict: dict[str, int]) -> PreprocessingFunction:
    def fn(recordings: list[Recording]):
        activities = set(itertools.chain(*[recording.activities for recording in recordings]))
        activity_str_to_id_dict = {v: idx for idx, v in enumerate(activities)}
        for recording in recordings:
            recording.activities.replace(activity_str_to_id_dict, inplace=True)

        empty_activity_dict.update(activity_str_to_id_dict)
        return recordings
    return fn
