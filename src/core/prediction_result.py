from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    y_train_true: Any
    y_train_pred: Any
    y_test_true: Any
    y_test_pred: Any
    y_train_encoded: Any = None
    y_test_encoded: Any = None
    train_proba: Any = None
    test_proba: Any = None
    model: Any = str
