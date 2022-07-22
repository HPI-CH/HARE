import numpy as np
import utils.settings as settings

def n_windows_per_activity_dict(y_test: np.ndarray) -> dict:
    """
    check if data is balanced!
    format: activity_name: n_windows

    returns {
        'walking': 23849,
        'running': 53,
        'sitting': 5,
    }
    """
    y_test_idx = np.argmax(y_test, axis=1)
    unique_values = np.unique(y_test_idx, return_counts=True)
    unique_counts = {}
    for i in range(len(unique_values[0])):
        activity_idx = unique_values[0][i]
        activity_name = settings.activity_initial_num_to_activity_str[activity_idx]
        unique_counts[activity_name] = unique_values[1][i]
    return unique_counts