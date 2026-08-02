import plotly.express as px


def attack_accuracy(attack_metrics):
    return px.line(
        attack_metrics,
        x="dataset",
        y="attack_acc",
        color="model",
        markers=True,
        title="Acurácia do ataque por dataset",
    )

def advantage(attack_metrics):
    return px.line(
        attack_metrics,
        x="dataset",
        y="advantage",
        facet_col="model",
        facet_col_wrap=2,
        markers=True,
    )

def build_average_advantage(attack_metrics):
    avg_advantage = (
        attack_metrics.groupby("model", as_index=False)["advantage"]
        .mean()
        .sort_values("advantage", ascending=False)
    )
    return avg_advantage