from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TaskConfig:

    task_type: str  
    target: str
