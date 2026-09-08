
import pandas as pd
import matplotlib.pyplot as plt


def plot_classifications_results(df):
    # ============================================================
    # PREPARAÇÃO
    # ============================================================

    # Remove caracteres '*' caso tenham vindo do CSV
    df.columns = df.columns.str.replace("*", "", regex=False)

    for column in [
        "test_acc",
        "validation_acc",
        "train_acc",
        "train_precision",
        "validation_precision",
        "test_precision",
        "train_recall",
        "validation_recall",
        "test_recall",
        "train_f1",
        "validation_f1",
        "test_f1",
        "generalization_gap",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")


    # Ordem desejada dos datasets
    dataset_order = [
        "baseline",
        "dp_eps_0.1",
        "dp_eps_0.5",
        "dp_eps_1.0",
        "dp_eps_2.0",
    ]

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=dataset_order,
        ordered=True,
    )

    df = df.sort_values("dataset")


    # Nome mais legível
    dataset_labels = {
        "baseline": "Baseline",
        "dp_eps_0.1": "ε = 0.1",
        "dp_eps_0.5": "ε = 0.5",
        "dp_eps_1.0": "ε = 1.0",
        "dp_eps_2.0": "ε = 2.0",
    }


    # ============================================================
    # 1. ACCURACY — TRAIN / VALIDATION / TEST
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["train_acc"],
        marker="o",
        label="Train",
    )

    plt.plot(
        df["dataset"],
        df["validation_acc"],
        marker="o",
        label="Validation",
    )

    plt.plot(
        df["dataset"],
        df["test_acc"],
        marker="o",
        label="Test",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.title("Accuracy por conjunto de dados")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 2. MÉTRICAS DE TESTE
    # ============================================================

    metrics = {
        "test_acc": "Accuracy",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1",
    }

    plt.figure(figsize=(10, 6))

    for column, label in metrics.items():
        plt.plot(
            df["dataset"],
            df[column],
            marker="o",
            label=label,
        )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Score")
    plt.xlabel("Dataset")
    plt.title("Métricas de teste")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 3. GENERALIZATION GAP
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["generalization_gap"],
        marker="o",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Generalization Gap (%)")
    plt.xlabel("Dataset")
    plt.title("Generalization Gap")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 4. PRECISION / RECALL / F1 — TESTE
    # ============================================================

    test_metrics = [
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    metric_labels = {
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1",
    }

    plt.figure(figsize=(10, 6))

    for metric in test_metrics:
        plt.plot(
            df["dataset"],
            df[metric],
            marker="o",
            label=metric_labels[metric],
        )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Score")
    plt.xlabel("Dataset")
    plt.title("Precision, Recall e F1 no conjunto de teste")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 5. DIFERENÇA EM RELAÇÃO À BASELINE
    # ============================================================

    baseline = df.loc[
        df["dataset"] == "baseline"
    ].iloc[0]

    privacy_df = df[
        df["dataset"] != "baseline"
    ].copy()

    privacy_df["accuracy_delta"] = (
        privacy_df["test_acc"] - baseline["test_acc"]
    )

    privacy_df["precision_delta"] = (
        privacy_df["test_precision"] - baseline["test_precision"]
    )

    privacy_df["recall_delta"] = (
        privacy_df["test_recall"] - baseline["test_recall"]
    )

    privacy_df["f1_delta"] = (
        privacy_df["test_f1"] - baseline["test_f1"]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        privacy_df["dataset"],
        privacy_df["accuracy_delta"],
        marker="o",
        label="Accuracy",
    )

    plt.plot(
        privacy_df["dataset"],
        privacy_df["precision_delta"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        privacy_df["dataset"],
        privacy_df["recall_delta"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        privacy_df["dataset"],
        privacy_df["f1_delta"],
        marker="o",
        label="F1",
    )

    plt.axhline(0, linestyle="--")

    plt.xticks(
        range(len(privacy_df)),
        [dataset_labels[x] for x in privacy_df["dataset"]],
    )

    plt.ylabel("Δ em relação à baseline")
    plt.xlabel("Dataset")
    plt.title("Variação das métricas em relação à baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 6. HEATMAP NUMÉRICO DAS MÉTRICAS
    # ============================================================

    metric_columns = [
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    heatmap_df = df[
        ["dataset"] + metric_columns
    ].set_index("dataset")

    heatmap_df.index = [
        dataset_labels[x]
        for x in heatmap_df.index
    ]

    plt.figure(figsize=(9, 5))

    plt.imshow(
        heatmap_df,
        aspect="auto",
    )

    plt.xticks(
        range(len(metric_columns)),
        ["Accuracy", "Precision", "Recall", "F1"],
    )

    plt.yticks(
        range(len(heatmap_df)),
    )

    plt.colorbar(label="Score")

    plt.title("Métricas de teste por nível de privacidade")

    for i in range(len(heatmap_df)):
        for j in range(len(metric_columns)):
            plt.text(
                j,
                i,
                f"{heatmap_df.iloc[i, j]:.3f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.show()