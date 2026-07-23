from statistics import mean

import pandas as pd


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()

    results = {}
    for key in keys:
        values = [metrics[key] for metrics in metrics_list]

        if isinstance(values[0], (int, float)):
            results[key] = round(mean(values), 3)

    return results


def aggregate_experiment_results(experiment_results):
    grouped_results = {}

    for result in experiment_results:
        key = (
            result.metadata["model_name"],
            result.metadata["dataset"],
        )
        grouped_results.setdefault(key, {"utility": [], "attack": []})
        grouped_results[key]["utility"].append(result.utility_metrics)
        grouped_results[key]["attack"].append(result.attack_metrics)

    utility_rows = []
    attack_rows = []

    for (model_name, dataset_name), metrics in grouped_results.items():
        utility_results = aggregate_metrics(metrics["utility"])
        attack_results = aggregate_metrics(metrics["attack"])

        utility_rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "test_mae": utility_results["test_mae"],
            "train_mae": utility_results["train_mae"],
            "generalization_gap_%": utility_results["generalization_gap_%"]
        })

        attack_rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "attack_acc": attack_results["attack_acc"],
            "member_acc": attack_results["member_acc"],
            "non_member_acc": attack_results["non_member_acc"],
            "advantage": attack_results["advantage"],
        })

    return pd.DataFrame(utility_rows), pd.DataFrame(attack_rows)
