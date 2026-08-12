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


def get_experiment_config():

    return ExperimentConfig(

        dataset_name="enem",

        dataset_version="enem_2025 - v-2026-07-21_23-19-59",

        sample_size=300_000,

        tasks=[
            TaskConfig(
                task_type="classification",
                target="TP_SEXO",
                active_models=[
                    ("XGBoost", run_xgboost),
                    ("Random Forest", run_random_forest),
                    ("Logistic Regression", run_linear_or_logistic_regression),
                ]
            ),
            TaskConfig(
                task_type="regression",
                target="Q005", #Incluindo você, quantas pessoas moram atualmente em sua residência?
                active_models=[
                    ("XGBoost", run_xgboost),
                    ("Random Forest", run_random_forest),
                    ("Linear Regression", run_linear_or_logistic_regression),
                ]
            ),
        ],

        preprocessing=PreprocessingConfig(
        
           categorical_columns=[
                "SG_UF_PROVA",
                "Q001",
                "Q002",
                "Q003",
                "Q004",
                "Q006",
                "Q008",
                "Q009",
                "Q010",
                "Q011",
                "Q012",
                "Q013",
                "Q014",
                "Q015",
                "Q016",
                "Q017",
                "Q018",
                "Q019",
                "Q020",
                "Q021",
                "Q022",
                "Q023",
           ],

             numerical_columns=[
                "TP_COR_RACA",
                "TP_FAIXA_ETARIA",
                "TP_ESTADO_CIVIL",
                "TP_ST_CONCLUSAO",
                "TP_ANO_CONCLUIU",
                "TP_ENSINO",
                "IN_TREINEIRO",
                "TP_NACIONALIDADE",
           ], 

        ),

        shadow_attack=ShadowAttackConfig(
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
        ),

    )