from dataclasses import dataclass

import pandas as pd

from .SensorFrame import SensorFrame


@dataclass
class Recording(SensorFrame):
    """
    Contains all data from one recording with the important context information
    """
    sensor_frame: pd.DataFrame
    time_frame: pd.Series

    activities: pd.Series
    subject: str

    name: str = None
