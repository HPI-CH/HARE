import os
import numpy as np

def create_text_metrics(path: str, y_test_pred: np.ndarray, y_test_true: np.ndarray, metric_funcs: list, file_name: str = None) -> None:
    """
    TODO: not working a the moment with more than one function, because the data gets changed (passed by reference)
    - make metric functions copy the data

    executes metric funcs  and writes the results in a text file with the provided path
    """
    if file_name is None:
        file_name = "metrics.txt"
    with open(os.path.join(path, file_name), "w") as f:
        for metric_func in metric_funcs:
            f.write(f"{metric_func.__name__}: {metric_func(y_test_pred, y_test_true)}\n")