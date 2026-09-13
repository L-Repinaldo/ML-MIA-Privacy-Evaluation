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

    train_f1: float | Any
    validation_f1: float | Any
    test_f1: float | Any



@dataclass
class ShadowModelMiaResult ():

    attack_acc: float | Any
    attack_f1: float | Any
    attack_precision: float | Any 

    member_acc_tpr: float | Any
    non_member_acc_tnr: float | Any

    advantage: float | Any 