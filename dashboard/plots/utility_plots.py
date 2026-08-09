import plotly.express as px
import plotly.graph_objects as go


def summary(utility_df):
    gap_summary = (
        _gap_summary(utility_df)
        .sort_values("generalization_gap")
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Modelo",
                        "Gap Médio (%)",
                        "Train MAE",
                        "Test MAE",
                    ]
                ),
                cells=dict(
                    values=[
                        gap_summary["model"],
                        gap_summary["generalization_gap"],
                        gap_summary["train_mae"],
                        gap_summary["test_mae"],
                    ]
                ),
            )
        ]
    )

    fig.update_layout(height=300)

    return fig


def mean_absolute_error(utility_df):
   
    df = utility_df.melt(
        id_vars=["model", "dataset", "epsilon"],
        value_vars=["train_mae", "test_mae"],
        var_name="split",
        value_name="mae",
    )

    fig = px.line(
        df,
        x="epsilon",
        y="mae",
        color="model",
        line_dash="split",
        markers=True,
        title="Train vs Test MAE",
        labels={
            "epsilon": "ε",
            "mae": "MAE",
            "split": "Conjunto",
        },
    )

    fig.update_layout(
        legend_title="Modelo / Conjunto",
        hovermode="x unified",
    )

    return fig


def utility_by_model(utility_df):
   
    df = utility_df.melt(
        id_vars=["model", "dataset", "epsilon"],
        value_vars=["train_mae", "test_mae"],
        var_name="split",
        value_name="mae",
    )

    fig = px.line(
        df,
        x="epsilon",
        y="mae",
        color="split",
        facet_col="model",
        facet_col_wrap=2,
        markers=True,
        title="Evolução da Utilidade por Modelo",
        labels={
            "epsilon": "ε",
            "mae": "MAE",
            "split": "Conjunto",
        },
    )

    fig.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )

    fig.update_layout(
        hovermode="x unified",
        legend_title="Conjunto",
    )

    return fig


def generalization_gap(utility_df):

    fig = px.line(
        utility_df,
        x="epsilon",
        y="generalization_gap_%",
        color="model",
        markers=True,
        title="Generalization Gap",
        labels={
            "epsilon": "ε",
            "generalization_gap_%": "Gap (%)",
        },
    )

    fig.update_layout(
        hovermode="x unified",
    )

    return fig


def _gap_summary(utility_df):
    return (
        utility_df
        .groupby("model", as_index=False)
        .agg(
            generalization_gap=("generalization_gap_%", "mean"),
            train_mae=("train_mae", "mean"),
            test_mae=("test_mae", "mean"),
        )
        .round(3)
    )