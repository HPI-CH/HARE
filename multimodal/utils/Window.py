import numpy as np
from dataclasses import dataclass

from utils.typing import assert_type


@dataclass
class Window:
    data_array: np.ndarray
    activity: int
    subject: str
    recording_index: int

    def __init__(
        self,
        sensor_array: np.ndarray,
        activity: int,
        subject: str,
        recording_index: int,
    ) -> None:
        assert_type(
            [
                (sensor_array, (np.ndarray, np.generic)),
                (activity, int),
                (subject, str),
                (recording_index, int),
            ]
        )
        self.data_array = sensor_array
        self.activity = activity
        self.subject = subject
        self.recording_index = recording_index
