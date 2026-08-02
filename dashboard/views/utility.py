
from builders import build_utility_df
from plots import utility_plots

import streamlit as st

def render_utility(utility_metrics):

    st.header("Utility Analysis")
    st.write(
        "Como a utilidade e a capacidade de generalização dos modelos "
        "variam sob diferentes níveis de Privacidade Diferencial?"
    )

    utility_df = build_utility_df.add_epsilon_column(utility_metrics)

    st.subheader("Test MAE x ε")
    st.plotly_chart(
        utility_plots.mean_absolute_error(utility_df),
        use_container_width=True,
    )

    st.subheader("Resumo")
    st.plotly_chart(
        utility_plots.summary(utility_df),
        use_container_width=True,
    )

    st.subheader("Utilidade por Modelo")
    st.plotly_chart(
        utility_plots.utility_by_model(utility_df),
        use_container_width=True,
    )