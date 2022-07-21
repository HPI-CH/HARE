from sonar_types import RecordingProcessingFunction, Recording


def split_first_n(n: int) -> RecordingProcessingFunction:
    """
    Splits a recording by taking the first n timesteps
    """

    def fn(recs: list[Recording]) -> list[Recording]:
        for rec in recs:
            if len(rec.sensor_frame) <= n:
                continue
            rec.sensor_frame = rec.sensor_frame[0:n]
            rec.time_frame = rec.time_frame[0:n]
            rec.activities = rec.activities[0:n]

        return recs

    return fn

