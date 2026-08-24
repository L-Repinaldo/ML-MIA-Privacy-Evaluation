from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DatasetConfig:

    dataset_name: str
    dataset_version: str
    data_sample_size: int
    data_random_state: int