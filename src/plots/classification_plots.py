import pandas as pd
import matplotlib.pyplot as plt


def plot_classifications_results(df):

    # PREPARAÇÃO

    for column in [
        "train_balanced_acc",
        "validation_balanced_acc",
        "test_balanced_acc",
        "train_precision",
        "validation_precision",
        "test_precision",
        "train_f1",
        "validation_f1",
        "test_f1",
        "generalization_gap",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    dataset_order = [
        "baseline",
        "dp_eps_0.05",
        "dp_eps_0.1",
        "dp_eps_0.5",
        "dp_eps_1.0",
        "dp_eps_2.0",
        "dp_eps_3.0",
    ]

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=dataset_order,
        ordered=True,
    )

    df = df.sort_values("dataset")

    dataset_labels = {
        "baseline": "Baseline",
        "dp_eps_0.05": "ε = 0.05",
        "dp_eps_0.1": "ε = 0.1",
        "dp_eps_0.5": "ε = 0.5",
        "dp_eps_1.0": "ε = 1.0",
        "dp_eps_2.0": "ε = 2.0",
        "dp_eps_3.0": "ε = 3.0",
    }

    x_labels = [
        dataset_labels[x]
        for x in df["dataset"]
    ]

    # 1. ACCURACY — TRAIN / VALIDATION / TEST

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_labels,
        df["train_balanced_acc"],
        marker="o",
        label="Train",
    )

    plt.plot(
        x_labels,
        df["validation_balanced_acc"],
        marker="o",
        label="Validation",
    )

    plt.plot(
        x_labels,
        df["test_balanced_acc"],
        marker="o",
        label="Test",
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.title("Accuracy por conjunto de dados")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. GENERALIZATION GAP TABLE

    generalization_df = df[
        [
            "dataset",
            "generalization_gap",
        ]
    ].copy()

    generalization_df["dataset"] = generalization_df["dataset"].map(
        dataset_labels
    )

    generalization_df = generalization_df.rename(
        columns={
            "dataset": "Dataset",
            "generalization_gap": "Generalization Gap (p.p.)",
        }
    )

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.axis("off")

    table = ax.table(
        cellText=generalization_df.round(3).values,
        colLabels=generalization_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    ax.set_title(
        "Generalization Gap por conjunto de dados",
        pad=20,
    )

    plt.tight_layout()
    plt.show()

    # 3. TRAIN SUMMARY TABLE

    summary_df = df[
        [
            "dataset",
            "train_balanced_acc",
            "train_precision",
            "train_f1",
        ]
    ].copy()

    summary_df["dataset"] = summary_df["dataset"].map(dataset_labels)

    summary_df = summary_df.rename(
        columns={
            "dataset": "Dataset",
            "train_balanced_acc": "Balanced Accuracy",
            "train_precision": "Precision",
            "train_f1": "F1",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis("off")

    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    ax.set_title(
        "Resumo das métricas de classificação — Treino",
        pad=20,
    )

    plt.tight_layout()
    plt.show()

    # 4. VALIDATION SUMMARY TABLE

    summary_df = df[
        [
            "dataset",
            "validation_balanced_acc",
            "validation_precision",
            "validation_f1",
        ]
    ].copy()

    summary_df["dataset"] = summary_df["dataset"].map(dataset_labels)

    summary_df = summary_df.rename(
        columns={
            "dataset": "Dataset",
            "validation_balanced_acc": "Balanced Accuracy",
            "validation_precision": "Precision",
            "validation_f1": "F1",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis("off")

    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    ax.set_title(
        "Resumo das métricas de classificação — Validação",
        pad=20,
    )

    plt.tight_layout()
    plt.show()

    # 5. TEST SUMMARY TABLE

    summary_df = df[
        [
            "dataset",
            "test_balanced_acc",
            "test_precision",
            "test_f1",
            "generalization_gap"
        ]
    ].copy()

    summary_df["dataset"] = summary_df["dataset"].map(dataset_labels)

    summary_df = summary_df.rename(
        columns={
            "dataset": "Dataset",
            "test_balanced_acc": "Balanced Accuracy",
            "test_precision": "Precision",
            "test_f1": "F1",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis("off")

    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    ax.set_title(
        "Resumo das métricas de classificação — Teste",
        pad=20,
    )

    plt.tight_layout()
    plt.show()