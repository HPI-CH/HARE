import numpy as np
from dataclasses import dataclass

from utils.typing import assert_type
from typing import Union

@dataclass
class Window:
    sensor_array: np.ndarray
    activity: int
    subject: str
    recording_index: int

    def __init__(
        self,
        sensor_array: np.ndarray,
        activity: int,
        subject: Union[str, int],
        recording_index: int,
    ) -> None:
        assert_type(
            [
                (sensor_array, (np.ndarray, np.generic)),
                (activity, int),
                (recording_index, int),
            ]
        )
        self.sensor_array = sensor_array
        self.activity = activity
        self.subject = subject
        self.recording_index = recording_index
