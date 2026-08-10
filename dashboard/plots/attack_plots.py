import plotly.express as px


def attack_accuracy(attack_metrics):
    return px.line(
        attack_metrics,
        x="dataset",
        y="attack_acc",
        color="model",
        line_dash="attack_type",
        markers=True,
        title="Acurácia do ataque por dataset (loss_confidence vs shadow_model)",
    )


def advantage(attack_metrics):
    return px.line(
        attack_metrics,
        x="dataset",
        y="advantage",
        facet_col="model",
        facet_col_wrap=2,
        line_dash="attack_type",
        markers=True,
    )


def build_average_advantage(attack_metrics):
    avg_advantage = (
        attack_metrics.groupby(["model", "attack_type"], as_index=False)[
            "advantage"
        ]
        .mean()
        .sort_values("advantage", ascending=False)
    )
    return avg_advantage