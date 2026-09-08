from dataclasses import dataclass

from src.core.dataset_config import DatasetConfig
from src.core.preprocessing_config import PreprocessingConfig
from src.core.task_config import TaskConfig

@dataclass
class UtilityExperimentConfig:

    dataset: DatasetConfig

    preprocessing: PreprocessingConfig

    task: TaskConfig

    active_datasets: list[str] | None = None
