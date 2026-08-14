import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plots.common import apply_epsilon_axis

SUMMARY_METRIC_LABELS = {
    "test_acc": "Test Accuracy",
    "test_precision": "Test Precision",
    "test_recall": "Test Recall",
    "test_f1": "Test F1",
    "test_mae": "Test MAE",
    "test_r2_score": "Test R²",
    "generalization_gap_%": "Generalization Gap (%)",
}

METRIC_LABELS = {
    "test_precision": "Precision",
    "test_recall": "Recall",
    "test_f1": "F1",
}


def accuracy(utility_df):
    fig = px.line(
        utility_df,
        x="epsilon",
        y="test_acc",
        color="model",
        markers=True,
        title="Accuracy no teste por ε",
        labels={
            "epsilon": "ε",
            "test_acc": "Accuracy",
        },
    )
    apply_epsilon_axis(fig)
    fig.update_layout(hovermode="x unified", legend_title="Modelo")
    return fig


def precision_recall_f1(utility_df):
    value_columns = [col for col in METRIC_LABELS if col in utility_df.columns]
    if not value_columns:
        return None

    id_vars = [col for col in ["model", "epsilon", "epsilon_label"] if col in utility_df.columns]

    df = utility_df.melt(
        id_vars=id_vars,
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    df["metric_label"] = df["metric"].map(METRIC_LABELS)

    fig = px.line(
        df,
        x="epsilon",
        y="value",
        color="model",
        facet_col="metric_label",
        facet_col_wrap=3,
        markers=True,
        title="Precision, Recall e F1 por ε",
        labels={
            "epsilon": "ε",
            "value": "Score",
        },
    )

    apply_epsilon_axis(fig)
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )
    fig.update_layout(hovermode="x unified", legend_title="Modelo")
    return fig


def mean_absolute_error(utility_df):

    fig = px.line(
        utility_df,
        x="epsilon",
        y="test_mae",
        color="model",
        markers=True,
        title="MAE no teste por ε",
        labels={
            "epsilon": "ε",
            "test_mae": "MAE (teste)",
        },
    )
    apply_epsilon_axis(fig)
    fig.update_layout(hovermode="x unified", legend_title="Modelo")
    return fig


def regression_r2(utility_df):

    fig = px.line(
        utility_df,
        x="epsilon",
        y="test_r2_score",
        color="model",
        markers=True,
        title="R² no teste por ε",
        labels={
            "epsilon": "ε",
            "test_r2_score": "R²",
        },
    )
    apply_epsilon_axis(fig)
    fig.update_layout(hovermode="x unified", legend_title="Modelo")
    return fig


def generalization_gap(utility_df):
    fig = px.line(
        utility_df,
        x="epsilon",
        y="generalization_gap_%",
        color="model",
        markers=True,
        title="Generalization Gap por ε",
        labels={
            "epsilon": "ε",
            "generalization_gap_%": "Gap (%)",
        },
    )
    apply_epsilon_axis(fig)
    fig.update_layout(hovermode="x unified", legend_title="Modelo")
    return fig


def summary(utility_df):
    metric_columns = [
        col for col in SUMMARY_METRIC_LABELS
        if col in utility_df.columns and utility_df[col].notna().any()
    ]

    aggregations = {
        col: (col, "mean")
        for col in metric_columns
    }

    summary_df = (
        utility_df
        .groupby("model", as_index=False)
        .agg(**aggregations)
        .round(3)
    )

    header = ["Modelo"] + [SUMMARY_METRIC_LABELS[col] for col in metric_columns]

    rows = [summary_df["model"].astype(str)]
    for col in metric_columns:
        rows.append([
            "" if pd.isna(value) else value
            for value in summary_df[col].tolist()
        ])

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header),
                cells=dict(values=rows),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig