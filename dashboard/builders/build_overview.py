import pandas as pd

from plots import common


def build(metadata, utility_metrics, attack_metrics):
    tradeoff_points = _build_tradeoff_points(
        utility_metrics=utility_metrics,
        attack_metrics=attack_metrics,
    )

    model_ranking = _build_model_ranking(tradeoff_points)

    return {
        "tradeoff_points": tradeoff_points,
        "model_ranking": model_ranking,
    }


def build_protocol(metadata, utility_metrics):
    targets = metadata.get("targets", {})
    targets_text = (
        ", ".join(f"{task}: {target}" for task, target in targets.items())
        if targets
        else "-"
    )

    shadow_config = metadata.get("shadow_attack", {})
    n_shadow_models = shadow_config.get("n_shadow_models", "-")

    return {
        "Timestamp": metadata.get("timestamp", "-"),
        "Seeds": ", ".join(map(str, metadata.get("seeds", []))),
        "Test sizes": ", ".join(map(str, metadata.get("test_sizes", []))),
        "Modelos": ", ".join(metadata.get("modelos", [])),
        "Datasets": ", ".join(
            sorted(utility_metrics["dataset"].unique())
        ),
        "Targets": targets_text,
        "Shadow models (ataque)": n_shadow_models,
    }


def _build_tradeoff_points(utility_metrics, attack_metrics):
    return common.build_tradeoff_points(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )


def _build_model_ranking(tradeoff_points):
    if tradeoff_points.empty:
        return pd.DataFrame(
            columns=[
                "model", "task_type", "attack_type",
                "utility_loss_mean", "advantage_mean", "attack_acc_mean",
            ]
        )

    # o ponto baseline (perda 0) é excluído da média para não diluir a
    # perda média de utilidade dos datasets DP.
    ranking_df = (
        tradeoff_points[tradeoff_points["dataset"] != "baseline"]
        .groupby(["model", "task_type", "attack_type"], as_index=False)
        .agg(
            utility_loss_mean=("utility_loss", "mean"),
            advantage_mean=("advantage", "mean"),
            attack_acc_mean=("attack_acc", "mean"),
        )
        .sort_values(
            by=["task_type", "attack_type", "advantage_mean", "utility_loss_mean"],
            ascending=[True, True, True, True],
        )
    )

    return ranking_df.round(3)