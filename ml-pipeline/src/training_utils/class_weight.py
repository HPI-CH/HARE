import numpy as np
from sklearn.utils import class_weight

from utils import error


def calculate_class_weight(*, y: np.ndarray=None, activities: list=None):
    """
    Calculates the class weight for given activity labels

    Pass either y (one-hot-encoded) or activities (list of labels)
    """
    if activities is None and y is None:
        error("Pass y or activities to calculate_class_weight, neither one was passed.")
        return None
    if activities is not None and y is not None:
        error("Pass either y or activities to calculate_class_weight, both were passed! Using activities")
    activities_total = activities
    if activities_total is None:
        activities_total = np.argmax(y, axis=1)

    # activities_total now contains an array of activities

    class_weight_calculated = class_weight.compute_class_weight('balanced', classes=np.unique(activities_total),
                                                                y=activities_total)
    class_weight_calculated = {i: v for i, v in enumerate(class_weight_calculated)}

    return class_weight_calculated