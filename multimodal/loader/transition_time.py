import pandas as pd
import numpy as np
import utils.settings as settings


def transition_time_cut(recordings: "list[Recording]", seconds_from_start = 2, seconds_from_end = 0) -> "list[Recording]":
    """
    1 - max 2 seconds at the end of every activity, to make windows cleaner    

    - the timestep_frequency needs to be set in the DATA_CONFIG (Opportunity dataset: 30 Hz)
    - will return the same number of recordings (no smooth transition anymore)
    - alternative: return much more recordings with only one activity each
    """

    # side effect implementation (modifies input data, no return required)
    # RAM performance decision to not deep copy and return new recordings
    timestep_frequency = settings.DATA_CONFIG.timestep_frequency

    n_timesteps_from_start = int(seconds_from_start * timestep_frequency)
    n_timesteps_from_end = int(seconds_from_end * timestep_frequency)

    for recording in recordings:
        activities = recording.activities.to_numpy()
        # change_idx = on this index new number
        inner_change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
        # add start and end
        change_idxs = np.concatenate(
            (np.array([0]), inner_change_idxs, np.array([len(activities)]))
        )
        cutting_start_end_tuples = []
        for i in range(len(change_idxs) - 1):
            cutting_start_end_tuples.append((change_idxs[i], change_idxs[i + 1]))

        # add n_timesteps_from_start from tuple[0] and substract n_timesteps_from_end from tuple[1]
        cut_tuple_idxs = lambda tuple: (tuple[0] + n_timesteps_from_start, tuple[1] - n_timesteps_from_end)
        cutting_start_end_tuples = list(map(cut_tuple_idxs, cutting_start_end_tuples))
        
        # filter out tuples doesnt make sense anymore
        has_window_len_bigger_0 = lambda tuple: tuple[1] - tuple[0] > 0
        cutting_start_end_tuples = list(filter(has_window_len_bigger_0, cutting_start_end_tuples))

        def cut_frame(frame):
            sub_frames = []
            for start, end in cutting_start_end_tuples:
                sub_frames.append(frame.iloc[start:end])
            return pd.concat(sub_frames).reset_index(drop=True)
        
        recording.time_frame = cut_frame(recording.time_frame)
        recording.activities = cut_frame(recording.activities)
        recording.sensor_frame = cut_frame(recording.sensor_frame)
    
    return recordings
