
from loaders import  artifact_loader

import streamlit as st



def select_artifact():
    artifacts = artifact_loader.list_artifacts()
    latest_artifact = artifact_loader.get_latest_artifact()

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
