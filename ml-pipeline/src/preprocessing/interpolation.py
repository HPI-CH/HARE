from sonar_types import Recording, PreprocessingFunction


def forward_fill() -> PreprocessingFunction:
    """
    Fills all nan values with the last not-nan value.
    """

    def fn(recordings: list[Recording]):
        for recording in recordings:
            recording.sensor_frame = recording.sensor_frame.fillna(method='ffill')

        return recordings

    return fn


def linear_interpolation() -> PreprocessingFunction:
    def _linear_interpolate(recordings: list[Recording]) -> list[Recording]:
        """
        df.interpolate() -> standard linear interpolation
        before:
                a	b	 c	  d
            0	1	4.0	 8.0  NaN
            1	2	NaN	 NaN  9.0
            2	3	6.0	 NaN  NaN
            3	4	6.0	 7.0  8.0
        after:
                a	b	c	        d
            0	1	4.0	8.000000	NaN
            1	2	5.0	7.666667	9.0
            2	3	6.0	7.333333	8.5
            3	4	6.0	7.000000	8.0
        -> other option fillna (taking the last availble value, duplicate it)
        -> what interpolation for what sensor makes semantic sense? quaternion? acceleration?
        """
        n_nan_values_before = 0
        n_nan_values_after = 0

        for recording in recordings:
            n_nan_values_before += recording.sensor_frame.isna().sum().sum()
            recording.sensor_frame = recording.sensor_frame.interpolate()
            n_nan_values_after += recording.sensor_frame.isna().sum().sum()

        print("number of NaN before interpolation", n_nan_values_before)
        print("number of NaN after interpolation", n_nan_values_after)

        return recordings

    return _linear_interpolate
