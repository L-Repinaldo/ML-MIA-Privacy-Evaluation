import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _add_epsilon_column(utility_metrics):
    epsilon_map = {
        "baseline": 0.0,
        "eps_0.1": 0.1,
        "eps_0.5": 0.5,
        "eps_1.0": 1.0,
        "eps_2.0": 2.0,
    }

    df = utility_metrics.copy()
    df["epsilon"] = df["dataset"].map(epsilon_map)

    return df


def render_utility(utility_metrics, attack_metrics, metadata):
    st.header("Utility Analysis")
    st.write(
        "Como a utilidade e a capacidade de generalização dos modelos "
        "variam sob diferentes níveis de Privacidade Diferencial?"
    )

    utility_df = _add_epsilon_column(utility_metrics)

    st.subheader("Test MAE × ε")

    fig_mae = px.line(
        utility_df,
        x="epsilon",
        y="test_mae",
        color="model",
        markers=True,
        title="Utilidade dos modelos sob diferentes níveis de ε",
        labels={
            "epsilon": "ε",
            "test_mae": "Test MAE",
        },
    )

    st.plotly_chart(fig_mae, use_container_width=True)

    gap_summary = (
        utility_df
        .groupby("model", as_index=False)
        .agg({
            "generalization_gap_%": "mean",
            "train_mae": "mean",
            "test_mae": "mean",
        }).round(2)
    )

    gap_summary = gap_summary.sort_values(
        by="generalization_gap_%",
        ascending=True
    )

    fig_summary = go.Figure(
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
                        gap_summary["generalization_gap_%"],
                        gap_summary["train_mae"],
                        gap_summary["test_mae"],
                    ]
                ),
            )
        ]
    )

    st.plotly_chart(fig_summary, use_container_width=True)

    st.subheader("Utilidade por Modelo")

    fig_mae_facets = px.line(
        utility_df,
        x="epsilon",
        y="test_mae",
        facet_col="model",
        facet_col_wrap=2,
        markers=True,
        title="Evolução da Utilidade por Modelo",
        labels={
            "epsilon": "ε",
            "test_mae": "Test MAE",
            "model": "Modelo",
        },
    )

    fig_mae_facets.for_each_annotation(
        lambda a: a.update(text=a.text.split("=")[-1])
    )

    st.plotly_chart(fig_mae_facets, use_container_width=True)