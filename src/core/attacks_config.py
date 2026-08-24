from dataclasses import dataclass, field
from src.core.models_spec_config import ModelSpec

@dataclass
class ShadowAttackConfig:

    enabled: bool = True

    n_shadow_models: int = 3
    member_fraction: float = 0.5
    attack_test_size: float = 0.3

    attack_features: list[str] = field(
        default_factory=lambda: [
            "probabilities",
            "confidence",
            "entropy",
            "loss",
        ]
    )

    attack_model: ModelSpec = ModelSpec(
                name="xgboost",
                model_type="xgboost_classifier",
                parameters={
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
                    "n_jobs": -1,
                    "random_state": 42,
                    "verbosity": 0,
                },
            )

