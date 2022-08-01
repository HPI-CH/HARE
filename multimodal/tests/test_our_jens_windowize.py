# pylint: disable=locally-disabled, import-error, line-too-long
# reason: the format should be like this for better readability, some imports are not accepted, 

import pandas as pd
from models.JensModel import JensModel
from utils.Recording import Recording

example_recordings = [
    Recording(
        # 30 timesteps, 3 features
        sensor_frame = pd.DataFrame([[1,11,111],[2,22,222],[3,33,333],[4,44,444],[5,55,555], [6,66,666], [7,77,777], [8,88,888], [9,99,999], [10,100,1000], [11, 111, 1111], [12, 222, 2222], [13, 333, 3333], [14, 444, 4444], [15, 555, 5555], [16, 666, 6666], [17, 777, 7777], [18, 888, 8888], [19, 999, 9999], [20, 1000, 10000], [21, 111, 1111], [22, 222, 2222], [23, 333, 3333], [24, 444, 4444], [25, 555, 5555], [26, 666, 6666], [27, 777, 7777], [28, 888, 8888], [29, 999, 9999], [30, 1000, 10000]]),
        time_frame = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        activities = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
        subject = "Bruder Jakob"
    ),
    Recording(
        # only one timestep
        sensor_frame = pd.DataFrame([[1, 11, 111]]),
        time_frame = pd.Series([1]),
        activities = pd.Series([0]), 
        subject = "Schwester Hortensie"
    )
]

"""
window_size = 5

rec_01:
    activities:
        (7*0) 2 windows, no waste;
        (3*1) 0 windows, 3 waste;
        (8*0) 2 windows, 1 waste,
        (1*1) 0 windows, 1 waste;
        (1*0) 0 windows, 1 waste;
        (9*1) 3 windows, no waste,
        (1*0) 0 windows, 1 waste
    -> total: 7 windows, 7 wasted_timesteps
    (4* 0-windows, 3* 1-window)

rec_02:
    activities:
        (1*0) 0 windows, 1 waste;
    -> total: 0 windows, 1 wasted_timestep

-> assert total: 7 windows, 8 wasted_timesteps, total 30 timesteps
(4* 0-windows, 3* 1-window)

n_total_timesteps = 30 (rec_01) + 1 (rec_02) = 31

"""

model = JensModel(window_size=25, n_features=50, n_outputs=5) # random init!!!
model.window_size = 5 # set manually, intitalization with 5 doesn't work
windows = model.windowize(example_recordings)

n_windows = 7
assert len(windows) == n_windows, "7 windows expected"
assert len(list(filter(lambda w: w.activity == 0, windows))) == 4, "4 0-windows expected" # if filter lambda true -> can stay
assert len(list(filter(lambda w: w.activity == 1, windows))) == 3, "3 1-windows expected"

window_start_idx = [1, 3, 11, 13, 21, 23, 25] # see ml-rainbow miro board for visualization of test
# (the first feature is equal to the time frame to check the cutting)
assert len(window_start_idx) == n_windows, "7 windows expected"
for i, window in enumerate(windows):
    assert window.data_array[0][0] == window_start_idx[i], f"window was cutted wrong, window {i} should be {window_start_idx[i]} at window.sensor_array[0][0]"

# manual check
for i, window in enumerate(windows):
    print(f"window {i+1}:\n", window.data_array, "\n\n")