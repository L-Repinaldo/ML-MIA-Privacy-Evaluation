from datetime import datetime
import gc

from artifacts import (
    build_artifact_metadata,
    build_experiment_id,
    persist_experiment_artifacts,
)

from src.core.dataset_preparation import prepare_dataset
from src.core.experiment_validation import validate_experiment_config
from src.core.runtime import configure_runtime

from src.data.dataset_registry import load_dataset_bundle
from src.data.sample_dataset import sample_dataset_bundle

from src.experiments.aggregation import aggregate_experiment_results
from src.experiments.run_experiment import run_machine_learning_experiments


class ExperimentalPipeline:

    def __init__(self, experiment_config):
        self.experiment_config = experiment_config

    def run(self):

        configure_runtime()

        validate_experiment_config(self.experiment_config)

        dataset_bundle = load_dataset_bundle(
            dataset_name=self.experiment_config.dataset_name,
            dataset_version=self.experiment_config.dataset_version,
        )

        dataset_bundle = sample_dataset_bundle(
            dataset_bundle,
            sample_size=self.experiment_config.sample_size,
            random_state=42,
        )

        experiment_results = []

        for dataset_name, df in zip(
            dataset_bundle["dataset_names"],
            dataset_bundle["datasets"],
        ):

            for task_config in self.experiment_config.tasks:

                for seed in self.experiment_config.seeds:

                    for test_size in self.experiment_config.test_sizes:

                        prepared_dataset = prepare_dataset(
                            name=dataset_name,
                            df=df,
                            target=task_config.target,
                            task_type=task_config.task_type,
                            preprocessing_config=self.experiment_config.preprocessing,
                            seed=seed,
                            test_size=test_size,
                        )

                        experiment_results.extend(
                            self._run_experiments(
                                prepared_dataset=prepared_dataset,
                                task_config=task_config,
                                seed=seed,
                                test_size=test_size,
                                target=task_config.target,
                            )
                        )

                        del prepared_dataset
                        gc.collect()

            del df
            gc.collect()

        df_utility, df_attack = aggregate_experiment_results(
            experiment_results
        )

        del experiment_results
        gc.collect()

        return self._persist_results(
            df_utility,
            df_attack,
            dataset_bundle,
        )

    def _run_experiments(
        self,
        *,
        prepared_dataset,
        task_config,
        seed,
        test_size,
        target,
    ) -> list:
        experiment_results = []

        for model_name, runner in task_config.active_models:

            experiment_results.extend(
                run_machine_learning_experiments(
                    model_runner=runner,
                    model_name=model_name,
                    prepared_dataset=prepared_dataset,
                    task_type=task_config.task_type,
                    seed=seed,
                    test_size=test_size,
                    target=target,
                    shadow_config=self.experiment_config.shadow_attack,
                )
            )

            gc.collect()

        return experiment_results

    def _persist_results(self, df_utility, df_attack, dataset_bundle):

        timestamp = datetime.now()

        experiment_id = build_experiment_id(timestamp)

        artifact_metadata = build_artifact_metadata(
            self.experiment_config,
            timestamp,
        )

        artifact_metadata.update(
            {
                "resolved_dataset_version": dataset_bundle["dataset_version"],
                "targets": {
                    task.task_type: task.target
                    for task in self.experiment_config.tasks
                },
            }
        )

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