from builders import build_utility_df

import streamlit as st


def render_task_selection(prepared_df, key):
    tasks = build_utility_df.available_tasks(prepared_df)
    if not tasks:
        tasks = [""]

    return st.radio(
        "Tarefa",
        tasks,
        format_func=lambda task: (
            build_utility_df.TASK_LABELS.get(task, task)
            if task
            else "(não informada)"
        ),
        key=key,
    )


def render_model_selection(prepared_df, task, key):
    models = build_utility_df.available_models(prepared_df, task)

    return st.multiselect(
        "Modelos",
        models,
        default=models,
        key=key,
        help="A lista reflete os modelos executados para a tarefa no artifact.",
    )


def render_dataset_selection(prepared_df, task, key):
    task_df = prepared_df[prepared_df["task_type"] == task]
    datasets = build_utility_df.ordered_datasets(task_df)

    return st.multiselect(
        "Datasets (privacidade)",
        datasets,
        default=datasets,
        format_func=build_utility_df.dataset_display_label,
        key=key,
    )