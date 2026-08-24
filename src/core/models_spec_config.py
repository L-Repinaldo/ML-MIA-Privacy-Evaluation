from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:

    name: str
    model_type: str
    parameters: dict[str, Any]