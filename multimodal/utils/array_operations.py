import numpy as np
import math

from utils import settings


def transform_to_subarrays(
    array: np.ndarray, subarray_size: int, stride_size: int
) -> 'list[np.ndarray]':
    """
    Transform an array into a list of subarrays.
    """
    start = 0
    array_size = array.shape[0]

    sub_windows = (
        start
        + np.expand_dims(np.arange(subarray_size), 0)
        +
        # Create a rightmost vector as [0, V, 2V, ...].
        np.expand_dims(np.arange(array_size - subarray_size, step=stride_size), 0).T
    )

    return array[sub_windows]


def shuffle_lists_equally(
    list_1: np.ndarray, list_2: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    assert len(list_1) == len(list_2), "Lists must have the same length, moron!"

    indices = np.random.permutation(list_1.shape[0])  # type: ignore
    return (list_1[indices], list_2[indices])


def split_list_by_percentage(
    list_to_split: list, percentage_to_split: float
) -> "tuple[list, list]":
    """
    Split a list into two sub lists
    the second sub list has the length 'len(list) * percentage_to_split'
    """
    split_index = math.floor(len(list_to_split) * percentage_to_split)
    return list_to_split[split_index:], list_to_split[:split_index]


def get_most_frequent_value_index(y: np.ndarray, axis: int):
    """
    Reduces an axis of an np.ndarray by bincount.
    Returns the most frequent value in numpy array row.
    """
    return np.argmax(np.apply_along_axis(np.bincount, 1, y), axis=axis)
