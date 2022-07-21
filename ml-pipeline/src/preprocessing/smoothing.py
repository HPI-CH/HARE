from sonar_types import Recording, PreprocessingFunction


def clamp(lower: int, upper: int) -> PreprocessingFunction:
    """
    Clamps / Clips all sensor values to a specific range given by upper & lower int
    """

    def clamp_fn(recording_list: list[Recording]):
        for recording in recording_list:
            recording.sensor_frame = recording.sensor_frame.clip(lower, upper)
        return recording_list

    return clamp_fn


def ewma_smoothing(span1=20, span2=10) -> PreprocessingFunction:
    """
    Applies Exponentially Weighted Moving Average smoothing to the sensor frames

    This is an easy way to get some smoothing and pretty quick to calculate

    https://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    """

    def fn(recording_list):
        for rec in recording_list:
            sf = rec.sensor_frame
            sf = sf.ewm(span=span1).mean()
            rec.sensor_frame = sf.ewm(span=span2).mean()
        return recording_list

    return fn
