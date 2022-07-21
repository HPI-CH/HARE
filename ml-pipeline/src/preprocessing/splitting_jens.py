import itertools

from sonar_types import Recording, Window, WindowizeFunction


def windowize_recording(window_size: int, recording: Recording) -> list[Window]:
    """
    Splits a single recording into windows where each window has only one activity

    Excess is ignored
    """
    windows = []
    recording_sensor_array = (recording.sensor_frame.to_numpy())
    activities = recording.activities.to_numpy()

    start = 0
    end = 0

    def last_start_stamp_not_reached(start):
        return start + window_size - 1 < len(recording_sensor_array)

    while last_start_stamp_not_reached(start):
        end = start + window_size - 1

        # has planned window the same activity in the beginning and the end?
        unique_activities_in_interval = set(activities[start: (end + 1)])
        if len(unique_activities_in_interval) == 1:
            window_sensor_array = recording_sensor_array[start: (end + 1), :]
            activity = activities[start]
            start += (window_size // 2)
            windows.append(
                Window(sensor_frame=window_sensor_array, activity=int(activity), subject=recording.subject)
            )

        else:
            while last_start_stamp_not_reached(start):
                if activities[start] != activities[start + 1]:
                    start += 1
                    break
                start += 1
    return windows


def windowize_by_activity_wasteful(window_size: int) -> WindowizeFunction:
    """
    Splits recordings into windows of window_size. Skips any possible windows which contain more than one activity.
    """

    def fn(recordings: list[Recording]):
        windows = [windowize_recording(window_size, recording) for recording in recordings]

        return list(itertools.chain.from_iterable(windows))

    return fn
