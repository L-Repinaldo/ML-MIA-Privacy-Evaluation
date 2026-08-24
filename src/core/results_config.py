from dataclasses import dataclass, field
from typing import Any

from src.core.splits_config import SplitConfig


@dataclass
class UtilityGlobalResult:

    split_config: SplitConfig
    model_name: str
    task_type: str


@dataclass
class UtilityResult:
    generalization_gap: float | Any
    task_type: str


@dataclass
class UtilityClassificationResult (UtilityResult):

    test_acc: float | Any
    train_acc: float | Any
    validation_acc: float | Any
    test_precision: float | Any
    test_recall: float | Any
    test_f1: float | Any


@dataclass
class UtilityRegressionResult (UtilityResult):

    train_abs_error: float
    test_abs_error: float
    test_mae: float
    train_mae: float
    test_r2: float
    train_r2: float
    validation_r2: float
