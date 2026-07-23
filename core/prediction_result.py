from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    y_train_true: Any
    y_train_pred: Any
    y_test_true: Any
    y_test_pred: Any
    model: Any
