from utils.typing import assert_type

import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import Union


@dataclass
class Recording:
    """
    our base data object
    Multilabel, so expects activity pd.Series

    Future: 
        - a dataclass creates the intializer automatically
            - consider only giving the attributes as class vars -> dataclass handles this
        - add self.recorder

    Refactoring idea:
    - subject should be a int!, add it to assert_type
    """

    def __init__(
        self,
        time_frame: pd.Series,
        activities: pd.Series,
        subject: str,
        recording_index: int,
        sensor_frame: pd.DataFrame = pd.DataFrame(),
        pose_frame: pd.DataFrame = pd.DataFrame(),
        recording_folder: str = None
    ) -> None:
        assert_type([
            (time_frame, pd.Series),
            (activities, pd.Series),
            (subject, str),
            (recording_index, int),
            (sensor_frame, pd.DataFrame),
            (pose_frame, pd.DataFrame),
            (recording_folder, str)
        ])
        
        print(activities[0])
        assert isinstance(activities[0], np.float64) or isinstance(activities[0], np.int64)
        assert sensor_frame.shape[0] == time_frame.shape[0] or sensor_frame.empty, "sensor_frame and time_frame have to have the same length"
        assert sensor_frame.shape[0] == activities.shape[0] or sensor_frame.empty, "sensor_frame and activities have to have the same length"   
        assert sensor_frame.shape[0] == pose_frame.shape[0] or sensor_frame.empty or pose_frame.empty, "sensor_frame and pose_frame have to have the same length"     
        self.sensor_frame = sensor_frame
        self.pose_frame = pose_frame
        self.time_frame = time_frame
        self.activities = activities
        self.subject = subject
        self.recording_index = recording_index
        self.recording_folder = recording_folder