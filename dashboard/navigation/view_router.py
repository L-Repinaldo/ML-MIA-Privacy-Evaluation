import streamlit as st

from components import sidebar
from loaders import artifact_loader

from views import (
    render_overview,
    render_utility,
    render_leakage,
)

VIEWS = {
    "Overview": render_overview,
    "Utility Analysis": render_utility,
    "Leakage Analysis": render_leakage,
}


def render_selected_view():

    st.title("ML Privacy Trade-off Explorer")

    st.sidebar.header("Experimento")

    artifact_dir = sidebar.select_artifact()

    selected_view = st.sidebar.radio(
        "Visão analítica",
        list(VIEWS.keys()),
    )

    if artifact_dir is None:
        st.info(
            "Execute `python main.py` para gerar artifacts experimentais."
        )
        return

    artifact = artifact_loader.load_artifact(artifact_dir)

    st.caption(f"Artifact: `{artifact['path'].name}`")

    if selected_view == "Overview":
        VIEWS[selected_view](
            utility_metrics=artifact["utility_metrics"],
            attack_metrics=artifact["attack_metrics"],
            metadata=artifact["metadata"],
        )
    elif selected_view == "Utility Analysis":
        VIEWS[selected_view](
            utility_metrics=artifact["utility_metrics"],
        )
    elif selected_view == "Leakage Analysis":
        VIEWS[selected_view](
            attack_metrics=artifact["attack_metrics"],
        )