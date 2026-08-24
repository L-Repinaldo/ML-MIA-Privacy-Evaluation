from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionResult:
    y_train_true: Any
    y_train_pred: Any

    y_validation_true: Any
    y_validation_pred: Any

    y_test_true: Any
    y_test_pred: Any

    train_proba: Any = None
    validation_proba: Any = None
    test_proba: Any = None

    y_train_encoded: Any = None
    y_validation_encoded: Any = None
    y_test_encoded: Any = None

    model: Any = str
