
from  plots import common


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

    return {
            "Timestamp": metadata.get("timestamp", "-"),
            "Seeds": ", ".join(map(str, metadata.get("seeds", []))),
            "Test sizes": ", ".join(map(str, metadata.get("test_sizes", []))),
            "Modelos": ", ".join(metadata.get("modelos", [])),
            "Datasets": ", ".join(
                sorted(utility_metrics["dataset"].unique())
            ),
        }

def _build_tradeoff_points(utility_metrics, attack_metrics):
    return common.build_tradeoff_points(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )

def _build_model_ranking(tradeoff_points):
    ranking_df = (
            tradeoff_points.groupby("model", as_index=False)
            .agg(
                utility_loss_mean=("utility_loss", "mean"),
                advantage_mean=("advantage", "mean"),
                attack_acc_mean=("attack_acc", "mean"),
            )
            .sort_values(
                by=["advantage_mean", "utility_loss_mean"],
                ascending=[True, True],
            )
        )
    
    return ranking_df.round(3)