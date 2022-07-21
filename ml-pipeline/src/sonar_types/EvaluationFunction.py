from typing import Callable, Any

import numpy as np

EvaluationFunction = Callable[[np.ndarray, np.ndarray], dict[str, Any]]
