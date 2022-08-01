from dataclasses import dataclass

from models.RainbowModel import RainbowModel


@dataclass
class MarkdownTestResult:
    """
    required data per test to create the markdown report
    """
    model: RainbowModel
    model_nickname: str

    # analytics
    n_windows_per_activity: dict
    accuracy: float
    average_failure_rate: float