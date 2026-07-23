from config import DATASET_VERSION
from core.experiment_config import ExperimentConfig
from core.experimental_pipeline import ExperimentalPipeline
from model import (
    run_random_forest,
    run_xgboost,
    run_linear_regression,
)


if __name__ == "__main__":
    experiment_config = ExperimentConfig(
        dataset_version=DATASET_VERSION,
        active_models=[
            ("XGBoost", run_xgboost),
            ("Random Forest", run_random_forest),
            ("Linear Regression", run_linear_regression),
        ],
    )

    pipeline = ExperimentalPipeline(experiment_config=experiment_config)
    pipeline.run()
