import pandas as pd
import streamlit as st

from visualization.common import build_tradeoff_points
from visualization.plots import plot_privacy_utility_tradeoff


def render_overview(utility_metrics, attack_metrics, metadata):
    st.header("Overview")
    st.write(
        "Resumo executivo do experimento e visão consolidada do trade-off "
        "entre utilidade e risco de inferência."
    )

    col_dataset, col_models, col_datasets = st.columns(3)

    col_dataset.metric(
        "Dataset version",
        metadata.get("dataset_version", "-"),
    )

    col_models.metric(
        "Modelos",
        len(metadata.get("modelos", [])),
    )

    col_datasets.metric(
        "Datasets avaliados",
        utility_metrics["dataset"].nunique(),
    )

    with st.expander("Protocolo Experimental"):
        protocol = {
            "Timestamp": metadata.get("timestamp", "-"),
            "Seeds": ", ".join(map(str, metadata.get("seeds", []))),
            "Test sizes": ", ".join(map(str, metadata.get("test_sizes", []))),
            "Modelos": ", ".join(metadata.get("modelos", [])),
            "Datasets": ", ".join(
                sorted(utility_metrics["dataset"].unique())
            ),
        }

        st.table(
            pd.DataFrame(
                protocol.items(),
                columns=["Campo", "Valor"],
            )
        )

    tradeoff_points = build_tradeoff_points(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )

    st.subheader("Privacy–Utility Trade-off")

    fig_tradeoff = plot_privacy_utility_tradeoff(
        utility_results=utility_metrics,
        attack_results=attack_metrics,
    )

    st.plotly_chart(fig_tradeoff, use_container_width=True,)

    st.caption(
        """
        Interpretação do Advantage:

        • ≈ 0.00 → comportamento aleatório

        • ≈ 0.03 → sinal fraco de vazamento

        • ≥ 0.05 → vazamento relevante
        """
    )

    st.subheader("Model Ranking")

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

    ranking_df = ranking_df.round(3)

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Trade-off Table")

    st.dataframe(
        tradeoff_points.round(3),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Utility Metrics"):
        st.dataframe(
            utility_metrics,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Attack Metrics"):
        st.dataframe(
            attack_metrics,
            use_container_width=True,
            hide_index=True,
        )