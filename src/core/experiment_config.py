from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PreprocessingConfig:

    categorical_columns: list[str]

    numerical_columns: list[str]

    drop_columns: list[str] = field(default_factory=list)

    one_hot_drop: str = "first"

    handle_unknown: str = "ignore"

    scale_numeric: bool = False

    impute_numeric: str | None = "median"

    impute_categorical: str | None = "most_frequent"

@dataclass
class ExperimentConfig:

    dataset_name: str

    dataset_version: str

    target: str

    sample_size: int

    preprocessing: PreprocessingConfig

    task_type: str = "regression"

    seeds: list[int] = field(
        default_factory=lambda: [42, 123, 999]
    )

    test_sizes: list[float] = field(
        default_factory=lambda: [0.2]
    )

    active_models: list[tuple[str, Callable]] = field(
        default_factory=list
    )

    active_datasets: list[str] | None = None

    @property
    def model_names(self) -> list[str]:

        return [
            name
            for name, _ in self.active_models
        ]