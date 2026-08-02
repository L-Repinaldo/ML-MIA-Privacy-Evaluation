
from builders import build_overview  

import streamlit as st
import pandas as pd



def summary_cards(metadata, utility_metrics):

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


def render_protocol(metadata, utility_metrics):
    with st.expander("Protocolo Experimental"):
            protocol = build_overview.build_protocol(metadata, utility_metrics)
    
            st.table(
                pd.DataFrame(
                    protocol.items(),
                    columns=["Campo", "Valor"],
                )
            )