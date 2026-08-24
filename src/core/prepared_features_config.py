from dataclasses import dataclass
from typing import Any

from sklearn.preprocessing import LabelEncoder


@dataclass
class PreparedFeatures:
    name: str
    target: str
    task_type: str

    X_train: Any
    X_validation: Any
    X_test: Any

    y_train: Any
    y_validation: Any
    y_test: Any

    target_encoder: LabelEncoder | None = None