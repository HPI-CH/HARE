from dataclasses import dataclass

import pandas as pd

from .SensorFrame import SensorFrame


@dataclass
class Window(SensorFrame):
    """
    Contains recording data, but splitted into a specific shape
    """
    sensor_frame: pd.DataFrame

    activity: int
    subject: int
