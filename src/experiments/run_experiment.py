from time import perf_counter

from tqdm import tqdm

from src.attacks import extract_attack_features
from src.core.experiment_result import ExperimentResult

from .attack_runner import evaluate_attack, run_attack
from .utility_runner import compute_utility_metrics


def run_machine_learning_experiments(
    *,
    model_runner,
    model_name,
    prepared_dataset,
    task_type,
    seed,
    test_size,
):
    stages = [
        "Training",
        "Metrics",
        "Attack",
    ]

    with tqdm(
        total=len(stages),
        desc=f"{model_name:18}",
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        leave=False,
    ) as progress:

        progress.set_postfix(
            dataset=prepared_dataset.name,
            seed=seed,
            test=f"{test_size:.2f}",
            stage="Training",
        )

        model_start = perf_counter()

        prediction_result = model_runner(
            prepared_dataset=prepared_dataset,
            task_type=task_type,
            seed=seed,
        )

        model_time = perf_counter() - model_start

        progress.update()

        progress.set_postfix(
            dataset=prepared_dataset.name,
            seed=seed,
            test=f"{test_size:.2f}",
            stage="Metrics",
        )

        utility_metrics = compute_utility_metrics(prediction_result)
        attack_features = extract_attack_features(utility_metrics)

        progress.update()

        progress.set_postfix(
            dataset=prepared_dataset.name,
            seed=seed,
            test=f"{test_size:.2f}",
            stage="Attack",
        )

        attack_start = perf_counter()

        attack_output = run_attack(attack_features)

        attack_time = perf_counter() - attack_start

        attack_metrics = evaluate_attack(attack_output)

        progress.update()

    print(
        f"{model_name:18}"
        f" | Dataset={prepared_dataset.name}"
        f" | Seed={seed}"
        f" | Test={test_size:.2f}"
        f" | Model={model_time:.2f}s"
        f" | Attack={attack_time:.2f}s"
    )

    return [
        ExperimentResult(
            utility_metrics=utility_metrics,
            attack_metrics=attack_metrics,
            metadata={
                "model_name": model_name,
                "dataset": prepared_dataset.name,
                "seed": seed,
                "test_size": test_size,
            },
        )
    ]