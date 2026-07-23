import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from streamlit_app.artifact_loader import (
    get_latest_artifact, 
    list_artifacts, 
    load_artifact)

from streamlit_app.views import (
    render_overview,
    render_utility,
    render_leakage,
)


VIEWS = {
    "Overview": render_overview,
    "Utility Analysis": render_utility,
    "Leakage Analysis": render_leakage,
}



def main():
    st.set_page_config(
        page_title="ML Privacy Trade-off Explorer",
        layout="wide",
    )
    st.title("ML Privacy Trade-off Explorer")

    st.sidebar.header("Experimento")
    artifact_dir = _select_artifact()
    selected_view = st.sidebar.radio("Visão analítica", list(VIEWS.keys()))

    if artifact_dir is None:
        st.info("Execute `python main.py` para gerar artifacts experimentais.")
        return

    artifact = load_artifact(artifact_dir)
    st.caption(f"Artifact: `{artifact['path'].name}`")

    VIEWS[selected_view](
        utility_metrics=artifact["utility_metrics"],
        attack_metrics=artifact["attack_metrics"],
        metadata=artifact["metadata"],
    )

def _select_artifact():
    artifacts = list_artifacts()
    latest_artifact = get_latest_artifact()

    if not artifacts:
        st.sidebar.warning("Nenhum artifact encontrado.")
        return None

    artifact_labels = [artifact.name for artifact in artifacts]
    latest_label = latest_artifact.name if latest_artifact is not None else artifact_labels[-1]

    use_latest = st.sidebar.checkbox("Usar artifact mais recente", value=True)
    if use_latest:
        st.sidebar.caption(f"Artifact selecionado: {latest_label}")
        return latest_artifact

    selected_label = st.sidebar.selectbox(
        "Selecionar artifact",
        artifact_labels,
        index=artifact_labels.index(latest_label),
    )
    return artifacts[artifact_labels.index(selected_label)]



if __name__ == "__main__":
    main()
