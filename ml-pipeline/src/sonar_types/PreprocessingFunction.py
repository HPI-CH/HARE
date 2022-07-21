from typing import Callable, Union, TypeVar

import numpy as np

from sonar_types import SensorFrame, Window, Recording

T = TypeVar("T", bound=SensorFrame)

XYSplit = tuple[np.array, np.array]
RecordingProcessingFunction = Callable[[list[T]], list[T]]
WindowizeFunction = Callable[[list[Recording]], list[Window]]
FinalSplitFunction = Callable[[list[T]], XYSplit]

PreprocessingFunction = Union[
    RecordingProcessingFunction,
    WindowizeFunction,
    FinalSplitFunction,
]
