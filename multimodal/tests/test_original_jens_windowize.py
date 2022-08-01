import pandas as pd
from models.JensModel import JensModel
from utils.Recording import Recording

def jens_windowize(dataCollection, window_size):
    """
    in the working version expects dataframe with these columns:

        ['IMU-BACK-accX', 'IMU-BACK-accY', 'IMU-BACK-accZ',
        'IMU-BACK-Quaternion1', 'IMU-BACK-Quaternion2', 'IMU-BACK-Quaternion3',
        'IMU-BACK-Quaternion4', 'IMU-RLA-accX', 'IMU-RLA-accY', 'IMU-RLA-accZ',
        'IMU-RLA-Quaternion1', 'IMU-RLA-Quaternion2', 'IMU-RLA-Quaternion3',
        'IMU-RLA-Quaternion4', 'IMU-LLA-accX', 'IMU-LLA-accY', 'IMU-LLA-accZ',
        'IMU-LLA-Quaternion1', 'IMU-LLA-Quaternion2', 'IMU-LLA-Quaternion3',
        'IMU-LLA-Quaternion4', 'IMU-L-SHOE-EuX', 'IMU-L-SHOE-EuY',
        'IMU-L-SHOE-EuZ', 'IMU-L-SHOE-Nav_Ax', 'IMU-L-SHOE-Nav_Ay',
        'IMU-L-SHOE-Nav_Az', 'IMU-L-SHOE-Body_Ax', 'IMU-L-SHOE-Body_Ay',
        'IMU-L-SHOE-Body_Az', 'IMU-L-SHOE-AngVelBodyFrameX',
        'IMU-L-SHOE-AngVelBodyFrameY', 'IMU-L-SHOE-AngVelBodyFrameZ',
        'IMU-L-SHOE-AngVelNavFrameX', 'IMU-L-SHOE-AngVelNavFrameY',
        'IMU-L-SHOE-AngVelNavFrameZ', 'IMU-R-SHOE-EuX', 'IMU-R-SHOE-EuY',
        'IMU-R-SHOE-EuZ', 'IMU-R-SHOE-Nav_Ax', 'IMU-R-SHOE-Nav_Ay',
        'IMU-R-SHOE-Nav_Az', 'IMU-R-SHOE-Body_Ax', 'IMU-R-SHOE-Body_Ay',
        'IMU-R-SHOE-Body_Az', 'IMU-R-SHOE-AngVelBodyFrameX',
        'IMU-R-SHOE-AngVelBodyFrameY', 'IMU-R-SHOE-AngVelBodyFrameZ',
        'IMU-R-SHOE-AngVelNavFrameX', 'IMU-R-SHOE-AngVelNavFrameY',
        'IMU-R-SHOE-AngVelNavFrameZ', 'Locomotion', 'HL_Activity',
        'file_index']
    
    test subset:

        ['IMU-BACK-accX', 'IMU-BACK-accY', 'IMU-BACK-accZ', 'Locomotion', 'HL_Activity', 'file_index']

    """
    # print(dataCollection.columns)
    HL_Activity_i = dataCollection.columns.get_loc("HL_Activity")
    # convert the data frame to numpy array
    data = dataCollection.to_numpy()
    # segment the data
    n = len(data)
    X = []
    y = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size - 1

        # has planned window the same activity in the beginning and the end, is from the same file in the beginning and the end
        # what if it changes back and forth?
        if (
            data[start][HL_Activity_i] == data[end][HL_Activity_i]
            and data[start][-1] == data[end][-1] # the last index is the file index
        ):

            # print(data[start:(end+1),0:(HL_Activity_i)])
            # first part time axis, second part sensor axis -> get window
            X.append(
                data[start : (end + 1), 0 : (HL_Activity_i - 1)] # data[timeaxis/row, featureaxis/column] data[1, 2] gives specific value, a:b gives you an interval
            )  # slice before locomotion
            y.append(data[start][HL_Activity_i])  # the first data point is enough
            start += window_size // 2  # 50% overlap!!!!!!!!!

        # if the frame contains different activities or from different objects, find the next start point
        # if there is a rest smaller than the window size -> skip (window small enough?)
        else:
            while start + window_size - 1 < n:
                # find the switch point -> the next start point
                # different file check missing! will come here again (little overhead)
                if data[start][HL_Activity_i] != data[start + 1][HL_Activity_i]:
                    break
                start += 1
            start += 1  # dirty fix for the missing 'different file' check?

    return X, y

# same example_recordings (first recording duplicated) from test_our_jens_windowize.py
# same window expectation!!!!

example_recordings = [
    Recording(
        # 30 timesteps, 3 features
        sensor_frame = pd.DataFrame([[1,11,111],[2,22,222],[3,33,333],[4,44,444],[5,55,555], [6,66,666], [7,77,777], [8,88,888], [9,99,999], [10,100,1000], [11, 111, 1111], [12, 222, 2222], [13, 333, 3333], [14, 444, 4444], [15, 555, 5555], [16, 666, 6666], [17, 777, 7777], [18, 888, 8888], [19, 999, 9999], [20, 1000, 10000], [21, 111, 1111], [22, 222, 2222], [23, 333, 3333], [24, 444, 4444], [25, 555, 5555], [26, 666, 6666], [27, 777, 7777], [28, 888, 8888], [29, 999, 9999], [30, 1000, 10000]]),
        time_frame = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        activities = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
        subject = "Bruder Jakob"
    ),
    Recording(
        # 30 timesteps, 3 features
        sensor_frame = pd.DataFrame([[1,11,111],[2,22,222],[3,33,333],[4,44,444],[5,55,555], [6,66,666], [7,77,777], [8,88,888], [9,99,999], [10,100,1000], [11, 111, 1111], [12, 222, 2222], [13, 333, 3333], [14, 444, 4444], [15, 555, 5555], [16, 666, 6666], [17, 777, 7777], [18, 888, 8888], [19, 999, 9999], [20, 1000, 10000], [21, 111, 1111], [22, 222, 2222], [23, 333, 3333], [24, 444, 4444], [25, 555, 5555], [26, 666, 6666], [27, 777, 7777], [28, 888, 8888], [29, 999, 9999], [30, 1000, 10000]]),
        time_frame = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        activities = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]),
        subject = "Bruder Jakob 2"
    ),
    Recording(
        # only one timestep
        sensor_frame = pd.DataFrame([[1, 11, 111]]),
        time_frame = pd.Series([1]),
        activities = pd.Series([0]), 
        subject = "Schwester Hortensie"
    ),
    
]

# adapt to to format expected by jens_windowize
dataCollection = pd.DataFrame()
for i, recording in enumerate(example_recordings):
    n_timesteps = len(recording.sensor_frame)

    locomotion_series = pd.Series([42] * n_timesteps) # will be filtered out
    file_index_series = pd.Series([i] * n_timesteps)

    recording_df = pd.concat([recording.sensor_frame, locomotion_series, recording.activities, file_index_series], axis=1)
    recording_df.columns = ['IMU-BACK-accX', 'IMU-BACK-accY', 'IMU-BACK-accZ', 'Locomotion', 'HL_Activity', 'file_index']

    dataCollection = pd.concat([dataCollection, recording_df])


# start test
window_size = 5
window_start_idx = [1, 3, 11, 13, 21, 23, 25] # from other test to confirm the right cutting

X, y = jens_windowize(dataCollection, window_size)
print(X)
print(y)


"""
result:
    got windows starting with [1, 3, 11, 13, 21, 23, 25]
    and a 19 window - change was to fast (not expected)

!! in the second recording the [1, 3] windows are missing! -> bad windowing!
-> but should not have a big impact on accuracy
"""