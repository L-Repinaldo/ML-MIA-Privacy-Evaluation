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
class TaskConfig:

    task_type: str  
    target: str
    active_models: list[tuple[str, Callable]] = field(default_factory=list)


@dataclass
class ShadowAttackConfig:
    """Configuração do Black-Box Membership Inference Attack baseado em
    Shadow Models.

    Este ataque é uma segunda implementação experimental e não altera em nada
    o MIA baseado em loss + confidence.
    """

    enabled: bool = True

    n_shadow_models: int = 3

    # Fração de cada shadow dataset destinada a membros (o restante é
    # shadow-non-member). O particionamento é sem sobreposição.
    member_fraction: float = 0.5

    # Fração das observações shadow usadas como validação interna do attack
    # model (diagnóstico). O attack model usa apenas a fração de treino.
    attack_test_size: float = 0.3

    # Conjunto de features black-box usadas pelo ataque.
    # classification -> ["probabilities", "confidence", "entropy", "loss"]
    # regression    -> prediction / absolute_error / squared_error
    attack_features: list[str] = field(
        default_factory=lambda: [
            "probabilities",
            "confidence",
            "entropy",
            "loss",
        ]
    )

    # Configuração do attack model (independente dos modelos-alvo).
    attack_model: dict = field(
        default_factory=lambda: {
            "type": "xgboost",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": 42,
            "verbosity": 0,
        }
    )


@dataclass
class ExperimentConfig:

    dataset_name: str

    dataset_version: str

    sample_size: int

    preprocessing: PreprocessingConfig

    tasks: list[TaskConfig] = field(default_factory=list)

    seeds: list[int] = field(
        default_factory=lambda: [42, 123, 999]
    )

    test_sizes: list[float] = field(
        default_factory=lambda: [0.2]
    )

    active_datasets: list[str] | None = None

    shadow_attack: ShadowAttackConfig | None = None

    @property
    def model_names(self) -> list[str]:

        return [
            name
            for task in self.tasks
            for name, _ in task.active_models
        ]