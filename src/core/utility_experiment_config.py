from dataclasses import dataclass, field
from typing import Callable

from src.core.dataset_config import DatasetConfig
from src.core.preprocessing_config import PreprocessingConfig
from src.core.task_config import TaskConfig

@dataclass
class UtilityExperimentConfig:

    dataset: DatasetConfig

    preprocessing: PreprocessingConfig

    tasks: list[TaskConfig] = field(default_factory=list)

    active_datasets: list[str] | None = None
