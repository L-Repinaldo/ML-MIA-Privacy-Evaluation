from src.core.experiment_config import (
    ExperimentConfig,
    PreprocessingConfig,
)

from src.model import (
    run_xgboost,
    run_random_forest,
    run_linear_regression,
)


def get_experiment_config():

    return ExperimentConfig(

        dataset_name="enem",

        dataset_version="enem_2025 - v-2026-07-21_23-19-59",

        target="Q005",

        sample_size=300_000,

        preprocessing=PreprocessingConfig(
        
           categorical_columns=[
                "TP_SEXO",
                "SG_UF_PROVA",
                "Q001",
                "Q002",
                "Q003",
                "Q004",
                "Q006",
                "Q007",
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

        active_models=[

            ("XGBoost", run_xgboost),

            ("Random Forest", run_random_forest),

            ("Linear Regression", run_linear_regression),

        ],

    )