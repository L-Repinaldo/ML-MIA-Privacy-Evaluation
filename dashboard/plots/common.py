from pathlib import Path

import numpy as np
import pandas as pd

from builders.build_utility_df import EPSILON_LABELS, EPSILON_MAP


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
        if str(dataset_name).startswith("dp_eps_")
    ]

    return sorted(
        datasets,
        key=lambda name: float(name.removeprefix("dp_eps_"))
    )


def apply_epsilon_axis(fig):
    """Aplica o eixo ε em ordem de privacidade (baseline → ε maior)."""
    tickvals = sorted(EPSILON_LABELS)
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=[EPSILON_LABELS[value] for value in tickvals],
        title="ε (intensidade de privacidade)",
    )
    return fig


def _utility_metric_for_task(utility_df, task):
    """Métrica de utilidade representativa da tarefa.

    classification → test_acc (maior = melhor)
    regression     → test_mae (menor = melhor)
    Fallback para artifacts legados sem task_type.
    """
    if task == "classification":
        return "test_acc", True
    if task == "regression":
        return "test_mae", False
    if "test_mae" in utility_df.columns and utility_df["test_mae"].notna().any():
        return "test_mae", False
    if "test_acc" in utility_df.columns and utility_df["test_acc"].notna().any():
        return "test_acc", True
    return None, None


def _relative_utility_loss(metric_value, baseline_value, higher_is_better):
    if (
        pd.isna(metric_value)
        or pd.isna(baseline_value)
        or baseline_value == 0
    ):
        return np.nan

    if higher_is_better:
        # perda relativa quando maior é melhor (ex.: accuracy)
        return (baseline_value - metric_value) / baseline_value
    # perda relativa quando menor é melhor (ex.: MAE)
    return (metric_value - baseline_value) / baseline_value


def build_tradeoff_points(utility_results, attack_results):
    """Monta pontos de trade-off privacidade × utilidade.

    Agrupa por (modelo, tarefa) e, para cada dataset DP, calcula a perda
    relativa de utilidade vs o baseline e associa as métricas de cada tipo
    de ataque MIA. Cada combinação (modelo, tarefa, dataset, attack_type)
    gera um ponto. Combinações sem ataque são omitidas (não inventadas).
    """
    utility_df = utility_results.copy()
    attack_df = attack_results.copy()

    if "task_type" not in utility_df.columns:
        utility_df["task_type"] = ""
        utility_df["target"] = ""
    if "task_type" not in attack_df.columns:
        attack_df["task_type"] = ""
        attack_df["target"] = ""
    if "attack_type" not in attack_df.columns:
        attack_df["attack_type"] = "loss_confidence"

    rows = []
    tasks = list(pd.unique(utility_df["task_type"]))

    for task in tasks:
        metric, higher_is_better = _utility_metric_for_task(utility_df, task)
        if metric is None:
            continue

        utility_task = utility_df[utility_df["task_type"] == task]

        for model_name in pd.unique(utility_task["model"]):
            utility_model = utility_task[utility_task["model"] == model_name]

            baseline_rows = utility_model[utility_model["dataset"] == "baseline"]
            baseline_rows = baseline_rows[baseline_rows[metric].notna()]
            if baseline_rows.empty:
                continue
            baseline_row = baseline_rows.iloc[0]
            baseline_value = baseline_row[metric]

            for _, utility_row in utility_model.iterrows():
                dataset_name = utility_row["dataset"]
                metric_value = utility_row[metric]
                epsilon = EPSILON_MAP.get(dataset_name, np.nan)

                for attack_type in pd.unique(attack_df["attack_type"]):
                    attack_rows = attack_df[
                        (attack_df["model"] == model_name)
                        & (attack_df["task_type"] == task)
                        & (attack_df["dataset"] == dataset_name)
                        & (attack_df["attack_type"] == attack_type)
                    ]
                    if attack_rows.empty:
                        continue
                    attack_row = attack_rows.iloc[0]

                    if dataset_name == "baseline":
                        utility_loss = 0.0
                    else:
                        utility_loss = _relative_utility_loss(
                            metric_value,
                            baseline_value,
                            higher_is_better,
                        )

                    rows.append({
                        "model": model_name,
                        "task_type": task,
                        "target": utility_row.get("target", ""),
                        "dataset": dataset_name,
                        "epsilon": epsilon,
                        "epsilon_label": EPSILON_LABELS.get(
                            epsilon, str(dataset_name)
                        ),
                        "attack_type": attack_type,
                        "utility_metric": metric,
                        "utility_metric_value": metric_value,
                        "baseline_metric_value": baseline_value,
                        "utility_loss": utility_loss,
                        "advantage": attack_row["advantage"],
                        "attack_acc": attack_row["attack_acc"],
                        "member_acc": attack_row["member_acc"],
                        "non_member_acc": attack_row["non_member_acc"],
                    })

    tradeoff = pd.DataFrame(rows)

    if not tradeoff.empty:
        tradeoff = tradeoff.sort_values(
            ["task_type", "model", "attack_type", "epsilon"],
            na_position="last",
        ).reset_index(drop=True)

    return tradeoff


def load_metric_artifacts(experiment_dir):
    artifact_dir = Path(experiment_dir)
    return (
        pd.read_csv(artifact_dir / "utility_metrics.csv"),
        pd.read_csv(artifact_dir / "attack_metrics.csv"),
    )