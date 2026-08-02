
from builders import build_overview  
from components import overview, raw_tables
from plots import tradeoff_plot

import streamlit as st


def render_overview(utility_metrics, attack_metrics, metadata):
    st.header("Overview")
    st.write(
        "Resumo executivo do experimento e visão consolidada do trade-off "
        "entre utilidade e risco de inferência."
    )

    overview.summary_cards(metadata, utility_metrics)

    overview.render_protocol(metadata, utility_metrics)

    overview_data= build_overview.build(metadata, utility_metrics, attack_metrics)

    st.subheader("Privacy-Utility Trade-off")

    st.plotly_chart(
         tradeoff_plot.plot_privacy_utility_tradeoff( overview_data["tradeoff_points"] )
         , use_container_width=True,
        )

    st.caption(
        """
        Interpretação do Advantage:

        • ≈ 0.00 → comportamento aleatório

        • ≈ 0.03 → sinal fraco de vazamento

        • ≥ 0.05 → vazamento relevante
        """
    )

    st.subheader("Model Ranking")


    st.dataframe(
        overview_data["model_ranking"].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Trade-off Table")

    st.dataframe(
        overview_data["tradeoff_points"].round(3),
        use_container_width=True,
        hide_index=True,
    )

    raw_tables.utility(utility_metrics)

    raw_tables.attack(attack_metrics)