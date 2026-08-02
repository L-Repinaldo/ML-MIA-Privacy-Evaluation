from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExperimentResult:
    utility_metrics: Dict[str, Any]
    attack_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
