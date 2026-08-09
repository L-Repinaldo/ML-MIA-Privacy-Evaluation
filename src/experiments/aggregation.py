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
        grouped_results.setdefault(key, {"utility": [], "attack": []})
        grouped_results[key]["utility"].append(result.utility_metrics)
        grouped_results[key]["attack"].append(result.attack_metrics)

    utility_rows = []
    attack_rows = []

    for (model_name, dataset_name, task_type, target), metrics in grouped_results.items():
        utility_results = aggregate_metrics(metrics["utility"])
        attack_results = aggregate_metrics(metrics["attack"])

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
            
        attack_rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "task_type": task_type,
            "target": target,
            "attack_acc": attack_results["attack_acc"],
            "attack_f1": attack_results["attack_f1"],
            "attack_precision": attack_results["attack_precision"],
            "attack_recall": attack_results["attack_recall"],
            "member_acc": attack_results["member_acc"],
            "non_member_acc": attack_results["non_member_acc"],
            "advantage": attack_results["advantage"],
        })

    return pd.DataFrame(utility_rows), pd.DataFrame(attack_rows)


def aggregate_metrics(metrics_list):
    keys = metrics_list[0].keys()

    results = {}
    for key in keys:
        values = [metrics[key] for metrics in metrics_list]

        if isinstance(values[0], (int, float)):
            results[key] = round(mean(values), 3)

    return results