from functools import reduce

from sonar_types import *


def apply_preprocessing(recordings: list[T], fns: list[PreprocessingFunction]) -> list:
    """
    Applies the given functions to the list of objects with sensor_frame in the order that they were passed.
    """
    return reduce(lambda o, func: func(o), fns, recordings)


def apply_split(recordings: list[Recording], fns: tuple[WindowizeFunction, FinalSplitFunction]) -> XYSplit:
    """
    Hacky way to get the typing correct. Is the same as apply_preprocessing but with slightly different types
    """
    r = fns[0](recordings)
    return fns[1](r)


def single_recording_preprocessing(fns: list[RecordingProcessingFunction]) -> Callable[[Recording], Recording]:
    def fn(recording: Recording):
        rec_list = apply_preprocessing([recording], fns)
        return rec_list[0] if len(rec_list) > 0 else None

    return fn
