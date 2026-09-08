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
    validation_acc: float | Any
    train_acc: float | Any

    train_precision: float | Any
    validation_precision: float | Any
    test_precision: float | Any

    train_recall: float | Any
    validation_recall: float | Any
    test_recall: float | Any

    train_f1: float | Any
    validation_f1: float | Any
    test_f1: float | Any


@dataclass
class UtilityRegressionResult (UtilityResult):

    train_mae: float
    validation_mae: float
    test_mae: float
    
    train_r2: float
    validation_r2: float  
    test_r2: float

    train_mse: float
    validation_mse: float
    test_mse: float

    train_mape: float
    validation_mape: float
    test_mape: float    



@dataclass
class ShadowModelMiaResult ():

    attack_acc: float | Any
    attack_f1: float | Any
    attack_precision: float | Any 
    attack_recall: float | Any
    member_acc: float | Any
    non_member_acc: float | Any
    advantage: float | Any 