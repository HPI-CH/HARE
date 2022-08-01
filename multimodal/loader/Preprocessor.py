from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from utils import settings
from utils.Recording import Recording
from utils.typing import assert_type


class Preprocessor:
    """
    Preprocessing templates (e.g. jens_preprocess) combine the private functions

    Refactoring Idea:
    - intitalize with functions that will be executed in the order
    - provide templates
    """

    def jens_preprocess(self, recordings: "list[Recording]") -> "list[Recording]":
        """
        1. _linear_interpolate
        """
        assert_type([(recordings[0], Recording)])

        # not needed? dataCollection = dataCollection.apply(pd.to_numeric, errors="coerce")  # data like 'k' (strings) will be converted to NaN
        recordings = self._linear_interpolate(recordings)
        return recordings
    
    def jens_preprocess_with_normalize(self, recordings: "list[Recording]") -> "list[Recording]":
        """
        1. _linear_interpolate
        2. _normalize_minmaxscaler
        """
        assert_type([(recordings[0], Recording)])

        # not needed? dataCollection = dataCollection.apply(pd.to_numeric, errors="coerce")  # data like 'k' (strings) will be converted to NaN
        recordings = self._linear_interpolate(recordings)
        recordings = self._normalize_minmaxscaler(recordings)
        return recordings


    def our_preprocess(self, recordings: "list[Recording]") -> "list[Recording]":
        """
        1. _interpolate_ffill
        2. _normalize
        """
        recordings = self._interpolate_ffill(recordings)
        recordings = self._normalize_standardscaler(recordings)
        return recordings

    # Preprocess-Library ------------------------------------------------------------

    def _linear_interpolate(self, recordings: "list[Recording]") -> "list[Recording]":
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
        assert_type([(recordings[0], Recording)])

        n_nan_values_before = 0
        n_nan_values_after = 0

        for recording in recordings:
            n_nan_values_before += recording.sensor_frame.isna().sum().sum()
            recording.sensor_frame = recording.sensor_frame.interpolate()
            n_nan_values_after += recording.sensor_frame.isna().sum().sum()

        print("number of NaN before interpolation", n_nan_values_before)
        print("number of NaN after interpolation", n_nan_values_after)

        return recordings


    def _interpolate_ffill(self, recordings: "list[Recording]") -> "list[Recording]":
        """
        the recordings have None values, this function interpolates them
        """
        assert_type([(recordings[0], Recording)])
        fill_method = "ffill"

        for recording in recordings:
            recording.sensor_frame = recording.sensor_frame.fillna(method=fill_method)

        return recordings
    
    def _normalize(self, recordings: 'list[Recording]', scaler_fn):
        """
        Normalizes the sensor values with the given scaler
        """
        assert_type([(recordings[0], Recording)])

        # First fit the scaler on all data
        scaler = scaler_fn()
        for recording in recordings:
            scaler.partial_fit(recording.sensor_frame)

        # Then apply normalization on each recording_frame
        for recording in recordings:
            transformed_array = scaler.transform(recording.sensor_frame)
            recording.sensor_frame = pd.DataFrame(
                transformed_array, columns=recording.sensor_frame.columns
            )
        return recordings


    def _normalize_minmaxscaler(self, recordings: "list[Recording]") -> "list[Recording]":
        return self._normalize(recordings, scaler_fn=MinMaxScaler)

    def _normalize_standardscaler(self, recordings: "list[Recording]") -> "list[Recording]":
        """
            Normalizes the sensor values to be in range 0 to 1
        """
        return self._normalize(recordings, scaler_fn=StandardScaler)


