from builders import build_overview, build_utility_df
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

    overview_data = build_overview.build(metadata, utility_metrics, attack_metrics)

    tradeoff_points = overview_data["tradeoff_points"]

    if tradeoff_points.empty:
        st.info("Não foi possível montar os pontos de trade-off para este artifact.")
    else:
        _render_tradeoff_section(tradeoff_points, overview_data["model_ranking"])

    raw_tables.utility(utility_metrics)

    raw_tables.attack(attack_metrics)


def _render_tradeoff_section(tradeoff_points, model_ranking):
    tasks = build_utility_df.available_tasks(tradeoff_points)
    if not tasks:
        tasks = [""]

    selected_task = st.radio(
        "Tarefa",
        tasks,
        format_func=lambda task: (
            build_utility_df.TASK_LABELS.get(task, task)
            if task
            else "(não informada)"
        ),
        key="overview_task",
    )

    attacks = sorted(
        tradeoff_points[tradeoff_points["task_type"] == selected_task][
            "attack_type"
        ].unique()
    )
    selected_attacks = st.multiselect(
        "Ataque MIA",
        attacks,
        default=attacks,
        #format_func=lambda attack: build_utility_df.ATTACK_LABELS.get(attack, attack),
        key="overview_attack",
    )
    if not selected_attacks:
        selected_attacks = attacks

    filtered = tradeoff_points[
        (tradeoff_points["task_type"] == selected_task)
        & (tradeoff_points["attack_type"].isin(selected_attacks))
    ]

    st.subheader("Privacy-Utility Trade-off")
    fig = tradeoff_plot.plot_privacy_utility_tradeoff(filtered)
    if fig is None:
        st.info("Sem pontos de trade-off para a seleção.")
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cada traço é a trajetória de um modelo, do baseline (perda de "
            "utilidade 0) até ε = 2.0. X usa a métrica de utilidade da tarefa "
            "(accuracy p/ classificação; MAE p/ regressão). Advantage **≈ 0** → "
            "ataque aleatório; **> 0** → vazamento detectável; **< 0** → ataque "
            "abaixo do aleatório. Valores reais são exibidos (sem truncamento)."
        )

    st.subheader("Model Ranking")
    ranking = model_ranking[
        (model_ranking["task_type"] == selected_task)
        & (model_ranking["attack_type"].isin(selected_attacks))
    ]
    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Trade-off Table")
    table_columns = [
        "model", "dataset", "epsilon_label", "attack_type",
        "utility_metric", "utility_metric_value", "utility_loss",
        "advantage", "attack_acc",
    ]
    table_columns = [col for col in table_columns if col in filtered.columns]
    st.dataframe(
        filtered[table_columns].round(3),
        use_container_width=True,
        hide_index=True,
    )