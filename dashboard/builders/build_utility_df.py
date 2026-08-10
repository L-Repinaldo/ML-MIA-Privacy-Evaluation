import pandas as pd

# Ordem canônica de privacidade: baseline vem primeiro e, depois, os
# datasets DP ordenados por epsilon crescente (0.1 → 2.0).
EPSILON_MAP = {
    "baseline": 0.0,
    "dp_eps_0.1": 0.1,
    "dp_eps_0.5": 0.5,
    "dp_eps_1.0": 1.0,
    "dp_eps_2.0": 2.0,
}

EPSILON_LABELS = {
    0.0: "Baseline",
    0.1: "ε=0.1",
    0.5: "ε=0.5",
    1.0: "ε=1.0",
    2.0: "ε=2.0",
}

TASK_LABELS = {
    "classification": "Classificação",
    "regression": "Regressão",
}

ATTACK_LABELS = {
    "loss_confidence": "Loss / Confidence",
    "shadow_model": "Shadow Model",
}

# Artifacts antigos não registravam o tipo de ataque; o único ataque
# executado era o de loss/confidence.
LEGACY_ATTACK_TYPE = "loss_confidence"


def add_epsilon_column(utility_metrics):
    """API pública preservada: adiciona a coluna numérica `epsilon`."""
    return _add_epsilon_column(utility_metrics)


def _add_epsilon_column(df):
    prepared = df.copy()
    prepared["epsilon"] = prepared["dataset"].map(EPSILON_MAP)
    return prepared


def add_epsilon_label_column(df):
    prepared = df.copy()
    prepared["epsilon_label"] = prepared["epsilon"].map(EPSILON_LABELS)
    prepared["epsilon_label"] = prepared["epsilon_label"].fillna(
        prepared["dataset"]
    )
    return prepared


def dataset_display_label(dataset_name):
    epsilon = EPSILON_MAP.get(dataset_name)
    if epsilon is not None:
        return EPSILON_LABELS.get(epsilon, str(dataset_name))
    return str(dataset_name)


def prepare_utility_df(utility_metrics):
    """Normaliza o schema de utility para o modelo conceitual atual.

    Adiciona `epsilon` (numérico), `epsilon_label` e colunas de
    compatibilidade (`task_type`, `target`) para artifacts legados.
    """
    df = _add_epsilon_column(utility_metrics)
    df = add_epsilon_label_column(df)
    if "task_type" not in df.columns:
        df["task_type"] = ""
    if "target" not in df.columns:
        df["target"] = ""
    return df


def prepare_attack_df(attack_metrics):
    """Normaliza o schema de ataque para o modelo conceitual atual.

    Garante `epsilon`, `epsilon_label`, `task_type`, `target`,
    `attack_type` e `attack_type_label`. Para artifacts legados (sem
    `attack_type`), assume o ataque de loss/confidence.
    """
    df = _add_epsilon_column(attack_metrics)
    df = add_epsilon_label_column(df)
    if "task_type" not in df.columns:
        df["task_type"] = ""
    if "target" not in df.columns:
        df["target"] = ""
    if "attack_type" not in df.columns:
        df["attack_type"] = LEGACY_ATTACK_TYPE
    df["attack_type_label"] = (
        df["attack_type"].map(ATTACK_LABELS).fillna(df["attack_type"])
    )
    return df


def available_tasks(prepared_df):
    """Tarefas presentes no artifact (ignora '' de artifacts legados)."""
    return [task for task in pd.unique(prepared_df["task_type"]) if task]


def available_models(prepared_df, task):
    """Modelos disponíveis para a tarefa selecionada.

    A lista é derivada dos próprios dados do artifact: modelos que não
    foram executados para a tarefa não aparecem.
    """
    df_task = prepared_df[prepared_df["task_type"] == task]
    return sorted(df_task["model"].unique())


def ordered_datasets(prepared_df):
    """Datasets em ordem de privacidade (baseline → ε maior)."""

    def _sort_key(dataset_name):
        epsilon = EPSILON_MAP.get(dataset_name)
        if epsilon is None:
            return (1, str(dataset_name))
        return (0, epsilon)

    return sorted(prepared_df["dataset"].unique(), key=_sort_key)