from dataclasses import dataclass

import pandas as pd


@dataclass
class SensorFrame:
    sensor_frame: pd.DataFrame

    def shape(self) -> tuple[int, int]:
        return self.sensor_frame.shape
