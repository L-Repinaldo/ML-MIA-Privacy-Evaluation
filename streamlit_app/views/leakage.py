import plotly.express as px
import streamlit as st


def render_leakage(utility_metrics, attack_metrics, metadata):
    st.header("Leakage Analysis")
    st.write("Como o risco de inferência varia por modelo e nível de privacidade?")

    st.subheader("Attack Accuracy")
    fig_attack_acc = px.line(
        attack_metrics,
        x="dataset",
        y="attack_acc",
        color="model",
        markers=True,
        title="Acurácia do ataque por dataset",
    )
    st.plotly_chart(fig_attack_acc, use_container_width=True)

    st.subheader("Advantage")
    fig_advantage = px.line(
        attack_metrics,
        x="dataset",
        y="advantage",
        facet_col="model",
        facet_col_wrap=2,
        markers=True,
    )

    fig_advantage.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="gray",
    )

    st.plotly_chart(fig_advantage, use_container_width=True)

    st.subheader("Average Advantage by Model")

    avg_advantage = (
        attack_metrics.groupby("model", as_index=False)["advantage"]
        .mean()
        .sort_values("advantage", ascending=False)
    )

    st.dataframe(
        avg_advantage,
        use_container_width=True,
        hide_index=True,
)