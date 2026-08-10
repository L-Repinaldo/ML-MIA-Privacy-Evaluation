from builders import build_utility_df
from components import filters
from plots import utility_plots

import streamlit as st


def render_utility(utility_metrics):
    st.header("Utility Analysis")
    st.write(
        "Como a utilidade e a capacidade de generalização dos modelos "
        "variam sob diferentes níveis de Privacidade Diferencial?"
    )
    st.caption(
        "As tarefas são tratadas separadamente: **classificação** "
        "(target TP_SEXO) e **regressão** (target Q005). "
        "Para regressão, MAE menor é melhor e R² maior é melhor."
    )

    utility_df = build_utility_df.prepare_utility_df(utility_metrics)

    selected_task = filters.render_task_selection(utility_df, key="utility_task")
    selected_models = filters.render_model_selection(utility_df, selected_task, key="utility_model")
    selected_datasets = filters.render_dataset_selection(utility_df, selected_task, key="utility_dataset")

    df = utility_df[
        (utility_df["task_type"] == selected_task)
        & (utility_df["model"].isin(selected_models))
        & (utility_df["dataset"].isin(selected_datasets))
    ].copy()

    if df.empty:
        st.info("Nenhum dado para a combinação selecionada.")
        return

    target_text = _target_label(df)

    if _is_classification(selected_task, df):
        st.subheader(f"Acurácia por ε {target_text}")
        st.plotly_chart(
            utility_plots.accuracy(df),
            use_container_width=True,
        )

        prf_fig = utility_plots.precision_recall_f1(df)
        if prf_fig is not None:
            st.subheader("Precision / Recall / F1")
            st.plotly_chart(prf_fig, use_container_width=True)
        elif "test_precision" not in df.columns:
            st.info("Precision/Recall/F1 não estão disponíveis neste artifact.")

        st.subheader("Generalization Gap")
        st.plotly_chart(
            utility_plots.generalization_gap(df),
            use_container_width=True,
        )
        st.caption(
            "Gap registrado no artifact: (accuracy de treino − accuracy "
            "de teste) × 100."
        )
    else:
        st.subheader(f"MAE por ε {target_text} — menor é melhor")
        st.plotly_chart(
            utility_plots.mean_absolute_error(df),
            use_container_width=True,
        )

        st.subheader("R² por ε — maior é melhor")
        st.plotly_chart(
            utility_plots.regression_r2(df),
            use_container_width=True,
        )

        st.subheader("Generalization Gap")
        st.plotly_chart(
            utility_plots.generalization_gap(df),
            use_container_width=True,
        )
        st.caption(
            "Gap registrado no artifact: ((MAE treino − MAE teste) / "
            "MAE treino) × 100."
        )

    st.subheader("Resumo por modelo")
    st.plotly_chart(
        utility_plots.summary(df),
        use_container_width=True,
    )


def _is_classification(selected_task, df):
    if selected_task == "classification":
        return True
    if selected_task == "regression":
        return False
    # artifact legado sem task_type: detecta pela métrica disponível
    return "test_acc" in df.columns and df["test_acc"].notna().any()


def _target_label(df):
    targets = df["target"].dropna().astype(str).unique()
    targets = [target for target in targets if target]
    if not targets:
        return ""
    return f"— target {', '.join(targets)}"