from builders import build_utility_df
from components import filters
from plots import attack_plots

import streamlit as st


def render_leakage(attack_metrics):
    st.header("Leakage Analysis")
    st.write(
        "Como o risco de inferência de membresia varia por modelo, "
        "nível de privacidade e tipo de ataque?"
    )
    st.caption(
        "Dois ataques são comparados explicitamente: **Loss/Confidence** "
        "(baseline) e **Shadow Model** (Black-Box MIA com Shadow Models)."
    )

    attack_df = build_utility_df.prepare_attack_df(attack_metrics)

    selected_task = filters.render_task_selection(attack_df, key="leakage_task")
    selected_models = filters.render_model_selection(attack_df, selected_task, key="leakage_model")
    selected_datasets = filters.render_dataset_selection(attack_df, selected_task, key="leakage_dataset")

    attack_types = sorted(
        attack_df[attack_df["task_type"] == selected_task]["attack_type"].unique()
    )
    selected_attacks = st.multiselect(
        "Tipo de ataque",
        attack_types,
        default=attack_types,
        #format_func=lambda attack: build_utility_df.ATTACK_LABELS.get(attack, attack),
        key="leakage_attack",
    )
    if not selected_attacks:
        selected_attacks = attack_types

    df = attack_df[
        (attack_df["task_type"] == selected_task)
        & (attack_df["model"].isin(selected_models))
        & (attack_df["dataset"].isin(selected_datasets))
        & (attack_df["attack_type"].isin(selected_attacks))
    ].copy()

    if df.empty:
        st.info("Nenhum dado de ataque para a combinação selecionada.")
        return

    target_text = _target_label(df)

    st.subheader(f"Attack Accuracy por ε {target_text}")
    st.plotly_chart(
        attack_plots.attack_accuracy(df),
        use_container_width=True,
    )
    st.caption("Accuracy ≈ 0.5 indica comportamento aleatório.")

    st.subheader("Advantage por ε")
    st.plotly_chart(
        attack_plots.advantage(df),
        use_container_width=True,
    )
    st.caption(
        "Interpretação (valores reais, sem truncamento): **≈ 0** → ataque "
        "próximo do aleatório; **> 0** → capacidade de distinguir membros; "
        "**< 0** → abaixo do aleatório. A linha tracejada marca advantage = 0."
    )

    mv_fig = attack_plots.member_vs_non_member(df)
    if mv_fig is not None:
        st.subheader("Member vs Non-member Accuracy")
        st.plotly_chart(mv_fig, use_container_width=True)
        st.caption("Accuracy 0.5 indica ausência de distinção do grupo.")

    f1_fig = attack_plots.attack_f1(df)
    if f1_fig is not None:
        st.subheader("F1 do ataque por ε")
        st.plotly_chart(f1_fig, use_container_width=True)

    st.subheader("Shadow Model vs Loss/Confidence")
    comparison_fig = attack_plots.attack_comparison(df)
    if comparison_fig is None:
        st.info(
            "A comparação ponto a ponto exige que ambos os ataques estejam "
            "selecionados para o mesmo modelo/target."
        )
    else:
        st.plotly_chart(comparison_fig, use_container_width=True)
        st.caption(
            "Cada ponto é uma combinação (modelo, dataset, tarefa). Pontos "
            "acima da linha y = x indicam que o Shadow Model detecta mais "
            "leakage que o ataque Loss/Confidence."
        )

    st.subheader("Average Advantage by Model")
    st.dataframe(
        attack_plots.build_average_advantage(df),
        hide_index=True,
        use_container_width=True,
    )

    _render_shadow_model_info(df)


def _render_shadow_model_info(df):
    shadow_rows = df[df["attack_type"] == "shadow_model"]
    if shadow_rows.empty:
        return

    required = {"n_shadow_models", "shadow_seed", "shadow_val_acc"}
    if not required.issubset(shadow_rows.columns):
        return

    info = (
        shadow_rows
        .groupby(["model", "dataset"], as_index=False)
        .agg(
            n_shadow_models=("n_shadow_models", "first"),
            shadow_seed=("shadow_seed", "first"),
            shadow_val_acc=("shadow_val_acc", "first"),
            attack_features=("attack_features", "first"),
        )
    )

    with st.expander("Shadow Model — parâmetros registrados no artifact"):
        st.dataframe(info, hide_index=True, use_container_width=True)
        st.caption(
            "Campos exclusivos do ataque Shadow Model. Para o ataque "
            "Loss/Confidence estes valores não se aplicam (NaN no artifact)."
        )


def _target_label(df):
    targets = df["target"].dropna().astype(str).unique()
    targets = [target for target in targets if target]
    if not targets:
        return ""
    return f"— target {', '.join(targets)}"