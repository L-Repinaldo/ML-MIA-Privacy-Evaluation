from datetime import datetime

from artifacts import (
    build_artifact_metadata,
    build_experiment_id,
    persist_experiment_artifacts,
)
from data.dataset_registry import load_registered_datasets
from experiments.aggregation import aggregate_experiment_results
from experiments.run_experiment import run_machine_learning_experiments


class ExperimentalPipeline:
    def __init__(self, experiment_config):
        self.experiment_config = experiment_config

    def run(self):
        self._configure_warnings()

        datasets, dataset_names = load_registered_datasets(
            dataset_version=self.experiment_config.dataset_version,
            active_datasets=self.experiment_config.active_datasets,
        )

        experiment_results = []

        for model_name, runner in self.experiment_config.active_models:
            print(f"\n{'='*40}")
            print(f"{model_name} execution")
            print(f"{'='*40}")

            experiment_results.extend(
                run_machine_learning_experiments(
                    model_runner=runner,
                    model_name=model_name,
                    datasets=datasets,
                    dataset_names=dataset_names,
                    seeds=self.experiment_config.seeds,
                    test_sizes=self.experiment_config.test_sizes,
                )
            )

        df_utility, df_attack = aggregate_experiment_results(experiment_results)

        timestamp = datetime.now()
        experiment_id = build_experiment_id(timestamp)
        artifact_metadata = build_artifact_metadata(self.experiment_config, timestamp)
        artifact_path = persist_experiment_artifacts(
            experiment_id=experiment_id,
            df_utility=df_utility,
            df_attack=df_attack,
            metadata=artifact_metadata,
        )

        return {
            "experiment_id": experiment_id,
            "artifact_path": artifact_path,
            "utility_metrics": df_utility,
            "attack_metrics": df_attack,
        }

    def _configure_warnings(self):
        import warnings

        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories in columns",
            category=UserWarning,
            module="sklearn.preprocessing._encoders"
        )
        warnings.filterwarnings(
            "ignore",
            message="`sklearn.utils.parallel.delayed` should be used"
        )
