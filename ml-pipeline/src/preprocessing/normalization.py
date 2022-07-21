import pandas as pd
from sklearn.preprocessing import StandardScaler

from sonar_types import Recording, PreprocessingFunction
from utils import verbose


def normalize(scaler=None):
    """
    Normalizes the sensor frame with the configured scaler. By default StandardScaler
    """
    if scaler is None:
        scaler = StandardScaler()

    def fn(recordings: list[Recording]) -> list[Recording]:
        # First fit the scaler on all data
        for id, recording in enumerate(recordings):
            verbose(f"Fit Rec #{id}", end='\r')
            scaler.partial_fit(recording.sensor_frame)

        # Then apply normalization on each recording_frame
        for id, recording in enumerate(recordings):
            verbose(f"Scale Rec #{id}", end='\r')
            transformed_array = scaler.transform(recording.sensor_frame)
            recording.sensor_frame = pd.DataFrame(
                transformed_array, columns=recording.sensor_frame.columns
            )
        verbose(f"Fitted and scaled all recordings.")
        return recordings

    return fn


def normalize_without_fit(scaler) -> PreprocessingFunction:
    def fn(recordings: list[Recording]) -> list[Recording]:
        # Then apply normalization on each recording_frame
        for recording in recordings:
            transformed_array = scaler.transform(recording.sensor_frame)
            recording.sensor_frame = pd.DataFrame(
                transformed_array, columns=recording.sensor_frame.columns
            )
        return recordings

    return fn
