
from plots import attack_plots

import streamlit as st


def render_leakage(attack_metrics):

    st.header("Leakage Analysis")
    st.write(
        "Como o risco de inferência varia por modelo e nível de privacidade?"
    )

    st.subheader("Attack Accuracy")
    st.plotly_chart(
        attack_plots.attack_accuracy(attack_metrics),
        use_container_width=True,
    )

    st.subheader("Advantage")
    st.plotly_chart(
        attack_plots.advantage(attack_metrics),
        use_container_width=True,
    )

    st.subheader("Average Advantage by Model")

    st.dataframe(
        attack_plots.build_average_advantage(attack_metrics),
        hide_index=True,
        use_container_width=True,
    )

