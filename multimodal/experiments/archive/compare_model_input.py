"""
compare_model_input.py
 - this test wants to monitor, how similar the windows are that get passed in jens pipeline vs. our rebuild of it
 - what does the model see? different windows, different preprocessing?
 - in both, the test_train_split is skipped
"""

import os
import random
import h5py
import numpy as np
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
from utils.folder_operations import new_saved_experiment_folder
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from models.RainbowModel import RainbowModel
from utils.progress_bar import print_progress_bar
from multiprocessing import Pool
from itertools import repeat


def our_X_y() -> (np.ndarray, np.ndarray):
    """
    our pipeline before the data gets passed into the model
    """
    # Load data
    recordings = load_opportunity_dataset(
        settings.opportunity_dataset_path
    )  # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)
    random.seed(1678978086101)
    random.shuffle(recordings)

    # Preprocessing
    recordings = Preprocessor().jens_preprocess(recordings)

    # Init
    model = JensModel(
        window_size=25,
        n_features=recordings[0].sensor_frame.shape[1],
        n_outputs=6,
        verbose=1,
        n_epochs=10,
    )  # random init, to get windowize func
    windows = model.windowize(
        recordings
    )  # we dont use jens convert as it expands a dimension, does categorical

    # !!!! From Rainbow Model convert !!!
    X = np.array(list(map(lambda window: window.data_array, windows)))
    y = np.array(list(map(lambda window: window.activity, windows)))

    return X, y


def jens_X_y() -> (np.ndarray, np.ndarray):
    """
    jens pipeline, before it gets passed into the model
    - execute data_processing.py before, this function just loads the windows, doesnt do the preprocessing again
    """
    preprocessed_h5_data_path = "research/jensOpportunityDeepL/hl_2.h5"
    file = h5py.File(preprocessed_h5_data_path, "r")
    X = file.get("inputs")
    y = file.get("labels")

    return X, y


def n_duplicate_windows(windows: np.ndarray) -> int:
    """
    Very uneffcient!!!
    - windows.shape: (n_windows, window_size, window_size, n_features)
    """
    unique_duplicate_windows = []

    def already_counted_as_duplicate(window):
        for unique_duplicate_window in unique_duplicate_windows:
            if np.array_equal(window, unique_duplicate_window):
                return True
        return False

    n_duplicate_windows = 0
    n_windows = windows.shape[0]
    for i, current_window in enumerate(windows):
        print_progress_bar(i, n_windows, prefix="n_duplicate_windows")
        found_duplicate = False
        for j in range(i + 1, n_windows):
            if np.array_equal(
                current_window, windows[j]
            ) and not already_counted_as_duplicate(current_window):
                found_duplicate = True
                n_duplicate_windows += 1
        if found_duplicate:
            unique_duplicate_windows.append(current_window)
    print_progress_bar(n_windows, n_windows, prefix="n_duplicate_windows")

    return n_duplicate_windows


def test_n_duplicate_windows():
    example_array = np.array([[1, 2], [5, 6], [7, 8], [1, 2], [1, 2], [1, 2], [7, 8]])
    assert 4 == n_duplicate_windows(
        example_array
    ), "n_duplicate_windows is working wrong"


def unique_intersection_np_axis_0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Takes around 1h to run for 50k windows
    - faster is n_elements_unique_intersection_np_axis_0 if you only need the len of the intersection array

    - a.shape: (n_windows, window_size, n_features)
    - b.shape: (n_windows, window_size, n_features)
    """

    a = np.unique(a, axis=0)
    b = np.unique(b, axis=0)

    windows_smaller_arr = a if len(a) < len(b) else b
    windows_bigger_arr = a if len(a) > len(b) else b

    intersection_array = np.empty(shape=tuple([0] + list(windows_smaller_arr[0].shape)))
    len_smaller_arr = len(windows_smaller_arr)
    for i, window in enumerate(windows_smaller_arr):
        print_progress_bar(i, len_smaller_arr, prefix="unique_intersection_np_axis_0")
        if window in windows_bigger_arr:
            intersection_array = np.append(
                np.expand_dims(window, axis=0), intersection_array, axis=0
            )
    print_progress_bar(
        len_smaller_arr, len_smaller_arr, prefix="unique_intersection_np_axis_0"
    )

    return intersection_array


def unique_intersection_np_axis_0_multiprocessing(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """
    !!! Not working !!! Multiprocessing not working
    """
    a = np.unique(a, axis=0)
    b = np.unique(b, axis=0)

    windows_smaller_arr = a if len(a) < len(b) else b
    windows_bigger_arr = a if len(a) > len(b) else b

    intersection_array = []
    print("unique_intersection_np_axis_0_multiprocessing")

    def in_bigger_window(window, windows_bigger_arr):
        print("Blub")
        return window if window in windows_bigger_arr else None

    pool = Pool()
    intersection_array = pool.map(
        in_bigger_window, zip(windows_smaller_arr, repeat(windows_bigger_arr))
    )
    # as often as you ask repeat(1) for a element in position it will return 1: [1, 1, 1, 1, 1, 1, ....] infinite times
    pool.close()
    pool.join()

    intersection_array = list(filter(lambda x: x is not None, intersection_array))

    return intersection_array


def n_elements_unique_intersection_np_axis_0(a: np.ndarray, b: np.ndarray) -> int:
    """
    A lot faster than to calculate the real intersection:

    Example with small numbers:
        a = [1, 4, 2, 13] # len = 4
        b = [1, 4, 9, 12, 25] # (len = 5)
        # a, b need to be unique!!!
        unique(concat(a, b)) = [1, 4, 2, 13, 9, 12, 25] # (len = 7)
        intersect(a, b) = [1, 4] # (len = 2) to expensive to call

        # Formular (fast to calculate)
        len(intersect(a, b)) = len(b) - n_elements_in_b_and_not_in_a
        len(intersect(a, b)) = len(b) - (len(unique(concat(a, b))) - len(a))
    """
    a = np.unique(a, axis=0)
    b = np.unique(b, axis=0)

    return len(b) - (len(np.unique(np.concatenate((a, b), axis=0), axis=0)) - len(a))


def test_unique_intersection_np_axis_0():
    """ 
    easy example to test the functions:
    - unique_intersection_np_axis_0
    - unique_intersection_np_axis_0_multiprocessing
    - n_elements_unique_intersection_np_axis_0

    """

    # window identifier on [0][0] => len(window_identifier) == n_windows
    to_window_identifier = lambda windows: list(
        map(lambda window: window[0][0], windows)
    )

    # example numpy array a of shape (7 (windows), 3 (window_size rows), 2 (n_features columns))
    a = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[19, 20], [21, 22], [23, 24]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
            [[7, 8], [9, 10], [11, 12]],
            [[19, 20], [21, 22], [23, 24]],
            [[7, 8], [9, 10], [11, 12]],
        ]
    )
    window_identifier_a = [1, 19, 7, 13, 7, 19, 7]
    assert window_identifier_a == to_window_identifier(a)

    # example numpy array b of shape (4 (windows), 3 (window_size rows), 2 (n_features columns))
    b = np.array(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[25, 26], [27, 28], [29, 30]],
        ]
    )
    window_identifier_b = [1, 7, 25]
    assert window_identifier_b == to_window_identifier(b)

    # Intersection
    py_list_unique_intersection = lambda list1, list2: list(
        filter(lambda x: x in set(list(list2)), set(list(list1)))
    )  # filter true: can stay
    window_identifier_unique_intersection = py_list_unique_intersection(
        window_identifier_a, window_identifier_b
    )

    # TEST 1 ----------------------------------------------------------------
    intersection_a_b = unique_intersection_np_axis_0(a, b)
    assert window_identifier_unique_intersection == sorted(
        to_window_identifier(intersection_a_b)
    ), "unique_intersection_np_axis_0 is not working"

    # TEST 2 ----------------------------------------------------------------
    # multiprocessing not working
    # intersection_a_b = unique_intersection_np_axis_0_multiprocessing(a, b)
    # assert window_identifier_unique_intersection == sorted(
    #     to_window_identifier(intersection_a_b)
    # ), "unique_intersection_np_axis_0_multiprocessing is not working"

    # TEST 3 ----------------------------------------------------------------
    # Faster calculation -> only len(intersection)
    assert n_elements_unique_intersection_np_axis_0(a, b) == len(intersection_a_b)


def save_data(data, path):
    """
    save the data in h5 format
    data should be dict
    path = path_to_folder + / + filename
    """
    f = h5py.File(path, "w")
    for key in data:
        f.create_dataset(key, data=data[key])
    f.close()


def load_data(path):
    """
    data is a dict
    """
    return h5py.File(path, "r")


def test_save_load_data():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}

    experiment_folder_path = new_saved_experiment_folder("save_load_data_test")
    path = os.path.join(experiment_folder_path, "test.h5")

    save_data(data, path)

    loaded_data_hd5 = load_data(path)
    loaded_data = {}
    for key in dict(loaded_data_hd5):
        loaded_data[key] = list(
            loaded_data_hd5[key]
        )  # for numpy array the convertion func is loaded_data_hd5.get(key)

    assert loaded_data == data, "save_data and load_data are not working"


def load_data_unique():
    # Load data
    X_our, y_our = our_X_y()  # our labels are categorical
    print("Shape of X_our:", X_our.shape)  # (53320, 25, 51)
    X_jens, y_jens = jens_X_y()
    print("Shape of X_jens:", X_jens.shape)  # (49484, 25, 51)

    # Unique
    X_our_unique = np.unique(X_our, axis=0)
    print(
        "Shape of X_our_unique:", X_our_unique.shape
    )  # (53187, 25, 51) we have 133 duplicate windows
    X_jens_unique = np.unique(X_jens, axis=0)
    print(
        "Shape of X_jens_set:", X_jens_unique.shape
    )  # (49484, 25, 51) no duplicate windows

    return X_our_unique, X_jens_unique


def save_intersection_windows_pipeline_compare() -> str:
    """
    takes 1h to execute, saves the data in a experiment folder
    """

    X_ours, X_jens = load_data_unique() # unique arrays!

    # Intersection
    X_intersection = unique_intersection_np_axis_0(
        X_ours, X_jens
    )  # 3,6 sec for 0.1% -> 1h
    print("Shape of X_intersection:", X_intersection.shape)

    # Save data
    experiment_folder_path = new_saved_experiment_folder(
        "save_intersection_windows_pipeline_compare"
    )
    path = os.path.join(experiment_folder_path, "test.h5")
    print("===> IMPORTANT: path to load intersection", path)
    save_data({"data": X_intersection}, path)

    return experiment_folder_path


def show_intersection_windows_pipeline_compare():
    """
    - execute save_intersection_windows_pipeline_compare() before (1h)
    - remember path to the data
    """
    path = "<experiment_folder_path>/test.h5"  # get from save_intersection_windows_pipeline_compare()

    loaded_data_hd5 = load_data(path)
    X_intersection = loaded_data_hd5.get(
        "data"
    )  # key from save_intersection_windows_pipeline_compare

    # Analysis
    print("Shape of X_intersection:", X_intersection.shape)
    # TODO: Debugger, see how how the data looks like


def show_n_intersection_windows_pipeline_compare():
    """
    This will not intersect the windows, as this takes 1h
    instead it uses a formular to calculate the size of the intersection array
    """

    X_ours, X_jens = load_data_unique() # unique arrays!

    print("Shape of X_ours:", X_ours.shape)  # (53320, 25, 51)
    print("Shape of X_jens:", X_jens.shape)  # (49484, 25, 51)
    print("len(intersection):", n_elements_unique_intersection_np_axis_0(X_ours, X_jens))

"""
Main

Tests to execute:
    - test_unique_intersection_np_axis_0()
    - test_n_duplicate_windows()
    - test_save_load_data()

Experiments to execute:
    Intersection:
    - save_intersection_windows_pipeline_compare() # takes 1h to execute, saves data
    - show_intersection_windows_pipeline_compare() # shows data

    len(Intersection):
    - show_n_intersection_windows_pipeline_compare

"""

settings.init()

show_n_intersection_windows_pipeline_compare()

