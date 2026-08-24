
from dataclasses import dataclass


@dataclass
class SplitConfig:

    seed: int | None = 42
    test_size: float | None = 0.2