from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple


@dataclass
class ExperimentConfig:
    dataset_version: str
    seeds: Sequence[int] = field(default_factory=lambda: [42, 123, 2026])
    test_sizes: Sequence[float] = field(default_factory=lambda: [0.2, 0.3])
    active_models: Sequence[Tuple[str, Any]] = field(default_factory=list)
    active_datasets: Optional[Sequence[str]] = None

    @property
    def model_names(self) -> List[str]:
        return [model_name for model_name, _ in self.active_models]
