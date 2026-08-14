from src.core.experiment_config import (
ExperimentConfig,
PreprocessingConfig,
ShadowAttackConfig,
TaskConfig,
)

from src.model import (
run_xgboost,
run_random_forest,
run_linear_or_logistic_regression,
)

CATEGORICAL_COLUMNS= [
    "Q001",
    "Q002",
    "Q003",
    "Q004",
    "Q008",
    "Q009",
    "Q010",
    "Q011",
    "Q012",
    "Q013",
    "Q018",
    "Q021",
    "Q022",
]

NUMERICAL_COLUMNS= [
    "TP_FAIXA_ETARIA",
    "TP_ANO_CONCLUIU",
]

COMMON_MODELS= [
    ("XGBoost", run_xgboost),
    ("Random Forest", run_random_forest),
]

CLASSIFICATION_MODELS= [
    *COMMON_MODELS,
    ( "Logistic Regression", run_linear_or_logistic_regression),
]

REGRESSION_MODELS= [
    *COMMON_MODELS,
    ("Linear Regression", run_linear_or_logistic_regression),
]

def get_experiment_config():

    # Run the experiment where both features and targets are affected by
    # differential privacy.
    return get_fully_private_config()

    # Run the experiment where only the features are affected by differential
    # privacy, while the targets remain unchanged.
    #return get_only_private_features_config()


def get_fully_private_config():

    return ExperimentConfig(

        dataset_name="enem",

        dataset_version="enem_2025 - v-2026-07-21_23-19-59",

        sample_size=100_000,

        tasks=[
            TaskConfig(
                task_type="classification",
                target="Q007", #Renda mensal familiar
                active_models= CLASSIFICATION_MODELS,
            ),
            TaskConfig(
                task_type="regression",
                target="TP_FAIXA_ETARIA",
                active_models= REGRESSION_MODELS,
            ),
        ],

        preprocessing=PreprocessingConfig(

            categorical_columns= CATEGORICAL_COLUMNS,

            numerical_columns= NUMERICAL_COLUMNS,

        ),

        shadow_attack=get_shadow_attack_config(),

    )


def get_only_private_features_config():

    return ExperimentConfig(

        dataset_name="enem",

        dataset_version="enem_2025 - v-2026-07-21_23-19-59",

        sample_size=100_000,

        tasks=[
            TaskConfig(
                task_type="classification",
                target="TP_SEXO",
                active_models= CLASSIFICATION_MODELS,
            ),
            TaskConfig(
                task_type="regression",
                target="Q005", #Quantas pessoas moram na residência
                active_models= REGRESSION_MODELS,
            ),
        ],

        preprocessing=PreprocessingConfig(
            categorical_columns= CATEGORICAL_COLUMNS,

            numerical_columns= NUMERICAL_COLUMNS,

        ),

        shadow_attack=get_shadow_attack_config(),

    )


def get_shadow_attack_config():

    return ShadowAttackConfig(
        enabled=True,
        n_shadow_models=5,
        member_fraction=0.5,
        attack_test_size=0.3,
        attack_features=[
            "probabilities",
            "confidence",
            "entropy",
            "loss",
        ],
        attack_model={
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
        },
    )