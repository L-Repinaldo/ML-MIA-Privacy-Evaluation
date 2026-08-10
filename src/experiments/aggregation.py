from statistics import mean

import pandas as pd


def aggregate_experiment_results(experiment_results):
    grouped_results = {}

    for result in experiment_results:
        key = (
            result.metadata["model_name"],
            result.metadata["dataset"],
            result.metadata["task_type"],
            result.metadata["target"],
        )
        grouped_results.setdefault(key, {"utility": [], "attack": {}})
        grouped_results[key]["utility"].append(result.utility_metrics)

        for attack_type, metrics in result.attack_metrics.items():
            grouped_results[key]["attack"].setdefault(attack_type, []).append(metrics)

    utility_rows = []
    attack_rows = []

    for key, metrics in grouped_results.items():
        model_name, dataset_name, task_type, target = key

        utility_results = aggregate_metrics(metrics["utility"])

        if task_type == "regression":
            utility_rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "target": target,
                        "test_mae": utility_results["test_mae"],
                        "test_r2_score": utility_results["test_r2"],
                        "generalization_gap_%": utility_results["generalization_gap_%"],
                    })
        elif task_type == "classification":
            utility_rows.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "target": target,
                        "test_acc": utility_results["test_acc"],
                        "test_precision": utility_results["test_precision"],
                        "test_recall": utility_results["test_recall"],
                        "test_f1": utility_results["test_f1"],
                        "generalization_gap_%": utility_results["generalization_gap_%"],
                    })

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        for attack_type, attack_metrics_list in metrics["attack"].items():
            attack_results = aggregate_metrics(attack_metrics_list)

            attack_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "task_type": task_type,
                "target": target,
                "attack_type": attack_type,
                "attack_acc": attack_results["attack_acc"],
                "attack_f1": attack_results["attack_f1"],
                "attack_precision": attack_results["attack_precision"],
                "attack_recall": attack_results["attack_recall"],
                "member_acc": attack_results["member_acc"],
                "non_member_acc": attack_results["non_member_acc"],
                "advantage": attack_results["advantage"],
                "n_shadow_models": attack_metrics_list[0].get("n_shadow_models"),
                "shadow_seed": attack_metrics_list[0].get("shadow_seed"),
                "shadow_val_acc": attack_results.get("shadow_val_acc"),
                "attack_features": attack_metrics_list[0].get("attack_features"),
            })

    return pd.DataFrame(utility_rows), pd.DataFrame(attack_rows)


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()

    results = {}
    for key in keys:
        values = [metrics[key] for metrics in metrics_list]

        if all(isinstance(value, (int, float)) for value in values):
            if all(value == values[0] for value in values):
                results[key] = values[0]
            else:
                results[key] = round(mean(values), 3)

    return results