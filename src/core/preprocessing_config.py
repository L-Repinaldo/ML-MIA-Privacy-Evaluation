
from dataclasses import dataclass, field


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