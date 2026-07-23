from pathlib import Path

import pandas as pd


def get_by_dataset(results_df, dataset_name):
    row = results_df[results_df["dataset"] == dataset_name]

    if row.empty:
        raise ValueError(f"Dataset {dataset_name} não encontrado nos resultados.")

    return row.iloc[0]


def get_by_model(results, model_name):
    df = results[results["model"] == model_name]

    if df.empty:
        raise ValueError(f"Modelo {model_name} não encontrado nos resultados.")

    return df


def get_epsilon_datasets(results_df):
    datasets = [
        dataset_name
        for dataset_name in results_df["dataset"].unique()
        if str(dataset_name).startswith("eps_")
    ]
    return sorted(datasets, key=lambda name: float(str(name).split("_")[1]))


def build_tradeoff_points(utility_results, attack_results):
    rows = []
    epsilon_datasets = get_epsilon_datasets(utility_results)

    for model_name in utility_results["model"].unique():
        df_utility = get_by_model(results=utility_results, model_name=model_name)
        df_attack = get_by_model(results=attack_results, model_name=model_name)

        baseline = get_by_dataset(results_df=df_utility, dataset_name="baseline")
        baseline_test_mae = baseline["test_mae"]

        for dataset_name in epsilon_datasets:
            utility_row = get_by_dataset(results_df=df_utility, dataset_name=dataset_name)
            attack_row = get_by_dataset(results_df=df_attack, dataset_name=dataset_name)

            epsilon = str(dataset_name).split("_")[1]
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "epsilon": epsilon,
                "utility_loss": abs(utility_row["test_mae"] - baseline_test_mae) / baseline_test_mae,
                "advantage": attack_row["advantage"],
                "test_mae": utility_row["test_mae"],
                "attack_acc": attack_row["attack_acc"],
            })

    return pd.DataFrame(rows)


def load_metric_artifacts(experiment_dir):
    artifact_dir = Path(experiment_dir)
    return (
        pd.read_csv(artifact_dir / "utility_metrics.csv"),
        pd.read_csv(artifact_dir / "attack_metrics.csv"),
    )
