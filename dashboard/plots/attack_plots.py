import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from builders.build_utility_df import ATTACK_LABELS
from plots.common import apply_epsilon_axis

ATTACK_METRIC_LABELS = {
    "member_acc": "Members",
    "non_member_acc": "Non-members",
}


def attack_accuracy(attack_metrics):
    fig = px.line(
        attack_metrics,
        x="epsilon",
        y="attack_acc",
        color="model",
        line_dash="attack_type_label",
        markers=True,
        title="Acurácia do ataque por ε",
        labels={
            "epsilon": "ε",
            "attack_acc": "Attack Accuracy",
        },
    )
    apply_epsilon_axis(fig)
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="aleatório (0.5)",
    )
    fig.update_layout(legend_title="Modelo / Ataque", hovermode="x unified")
    return fig


def attack_f1(attack_metrics):
    if "attack_f1" not in attack_metrics.columns:
        return None

    fig = px.line(
        attack_metrics,
        x="epsilon",
        y="attack_f1",
        color="model",
        line_dash="attack_type_label",
        markers=True,
        title="F1 do ataque por ε",
        labels={
            "epsilon": "ε",
            "attack_f1": "Attack F1",
        },
    )
    apply_epsilon_axis(fig)
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="gray",
        annotation_text="aleatório (0.5)",
    )
    fig.update_layout(legend_title="Modelo / Ataque", hovermode="x unified")
    return fig


def advantage(attack_metrics):
    fig = px.line(
        attack_metrics,
        x="epsilon",
        y="advantage",
        color="model",
        line_dash="attack_type_label",
        facet_col="attack_type_label",
        facet_col_wrap=2,
        markers=True,
        title="Advantage do ataque por ε",
        labels={
            "epsilon": "ε",
            "advantage": "Advantage",
        },
    )
    apply_epsilon_axis(fig)
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    fig.update_layout(legend_title="Modelo / Ataque", hovermode="x unified")
    return fig


def member_vs_non_member(attack_metrics):
    value_columns = [
        col for col in ATTACK_METRIC_LABELS if col in attack_metrics.columns
    ]
    if not value_columns:
        return None

    id_vars = [
        col for col in [
            "model", "epsilon", "epsilon_label", "attack_type_label"
        ]
        if col in attack_metrics.columns
    ]

    df = attack_metrics.melt(
        id_vars=id_vars,
        value_vars=value_columns,
        var_name="group",
        value_name="accuracy",
    )
    df["group_label"] = df["group"].map(ATTACK_METRIC_LABELS)

    fig = px.line(
        df,
        x="epsilon",
        y="accuracy",
        color="model",
        line_dash="attack_type_label",
        facet_col="group_label",
        facet_col_wrap=2,
        markers=True,
        title="Member vs Non-member Accuracy por ε",
        labels={
            "epsilon": "ε",
            "accuracy": "Accuracy",
        },
    )
    apply_epsilon_axis(fig)
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.update_layout(legend_title="Modelo / Ataque", hovermode="x unified")
    return fig


def attack_comparison(attack_metrics):

    if "attack_type" not in attack_metrics.columns:
        return None

    unique_attacks = attack_metrics["attack_type"].dropna().unique()
    if not {"loss_confidence", "shadow_model"}.issubset(unique_attacks):
        return None

    pivot = attack_metrics.pivot_table(
        index=["model", "dataset", "epsilon", "epsilon_label"],
        columns="attack_type",
        values="advantage",
    ).reset_index()

    if pivot["loss_confidence"].isna().all() or pivot["shadow_model"].isna().all():
        return None

    fig = px.scatter(
        pivot,
        x="loss_confidence",
        y="shadow_model",
        color="model",
        hover_data=["epsilon_label", "dataset"],
        labels={
            "loss_confidence": "Advantage — Loss / Confidence",
            "shadow_model": "Advantage — Shadow Model",
        },
        title="Advantage: Shadow Model vs Loss/Confidence",
    )

    values = pd.concat(
        [pivot["loss_confidence"], pivot["shadow_model"]]
    ).dropna()
    vmin, vmax = float(values.min()), float(values.max())
    padding = (vmax - vmin) * 0.1 if vmax > vmin else 0.005

    fig.add_trace(
        go.Scatter(
            x=[vmin - padding, vmax + padding],
            y=[vmin - padding, vmax + padding],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="y = x (igual)",
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray",
        annotation_text="aleatório",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        template="plotly_white",
        legend_title="Modelo",
        hovermode="closest",
        xaxis_tickformat=".3f",
        yaxis_tickformat=".3f",
    )
    return fig


def build_average_advantage(attack_metrics):
    avg_advantage = (
        attack_metrics
        .groupby(["model", "attack_type"], as_index=False)["advantage"]
        .mean()
        .sort_values("advantage", ascending=False)
        .round(3)
    )
    avg_advantage["attack_type"] = avg_advantage["attack_type"].map(
        lambda attack: ATTACK_LABELS.get(attack, attack)
    )
    return avg_advantage