

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_results(df):

        
    # Remove os "*" que vieram junto aos nomes/valores
    df.columns = df.columns.str.replace("*", "", regex=False)

    for col in df.columns:
        if col not in ["experiment_id", "dataset", "task_type",
                    "target", "model", "model_type"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("*", "", regex=False),
                errors="coerce"
            )

    dataset_order = [
        "baseline",
        "dp_eps_0.1",
        "dp_eps_0.5",
        "dp_eps_1.0",
        "dp_eps_2.0"
    ]

    dataset_labels = {
        "baseline": "Baseline",
        "dp_eps_0.1": "ε = 0.1",
        "dp_eps_0.5": "ε = 0.5",
        "dp_eps_1.0": "ε = 1.0",
        "dp_eps_2.0": "ε = 2.0"
    }

    df["dataset"] = pd.Categorical(
        df["dataset"],
        categories=dataset_order,
        ordered=True
    )

    df = df.sort_values("dataset")

    labels = [dataset_labels[x] for x in dataset_order]


    # ============================================================
    # 1. MAE — TRAIN / VALIDATION / TEST
    # ============================================================

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.25

    plt.bar(x - width, df["train_mae"], width, label="Train")
    plt.bar(x, df["validation_mae"], width, label="Validation")
    plt.bar(x + width, df["test_mae"], width, label="Test")

    plt.xticks(x, labels)
    plt.ylabel("MAE")
    plt.xlabel("Dataset")
    plt.title("MAE por dataset")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 2. R² — TRAIN / VALIDATION / TEST
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, df["train_r2"], width, label="Train")
    plt.bar(x, df["validation_r2"], width, label="Validation")
    plt.bar(x + width, df["test_r2"], width, label="Test")

    plt.xticks(x, labels)
    plt.ylabel("R²")
    plt.xlabel("Dataset")
    plt.title("R² por dataset")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 3. MSE — TRAIN / VALIDATION / TEST
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, df["train_mse"], width, label="Train")
    plt.bar(x, df["validation_mse"], width, label="Validation")
    plt.bar(x + width, df["test_mse"], width, label="Test")

    plt.xticks(x, labels)
    plt.ylabel("MSE")
    plt.xlabel("Dataset")
    plt.title("MSE por dataset")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 4. MAPE — TRAIN / VALIDATION / TEST
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, df["train_mape"], width, label="Train")
    plt.bar(x, df["validation_mape"], width, label="Validation")
    plt.bar(x + width, df["test_mape"], width, label="Test")

    plt.xticks(x, labels)
    plt.ylabel("MAPE")
    plt.xlabel("Dataset")
    plt.title("MAPE por dataset")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 5. COMPARAÇÃO DAS MÉTRICAS DE TESTE
    # ============================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    test_metrics = {
        "test_mae": "Test MAE",
        "test_r2": "Test R²",
        "test_mse": "Test MSE",
        "test_mape": "Test MAPE"
    }

    for ax, (metric, title) in zip(axes.flat, test_metrics.items()):

        values = df[metric]

        bars = ax.bar(labels, values)

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.suptitle("Métricas de teste", fontsize=15)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 6. GENERALIZATION GAP
    # ============================================================

    plt.figure(figsize=(10, 6))

    bars = plt.bar(labels, df["generalization_gap"])

    plt.axhline(0, linewidth=1)

    plt.ylabel("Generalization Gap (%)")
    plt.xlabel("Dataset")
    plt.title("Generalization Gap")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, df["generalization_gap"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 7. DELTA EM RELAÇÃO AO BASELINE
    # ============================================================

    baseline = df[df["dataset"] == "baseline"].iloc[0]

    delta_metrics = [
        "test_mae",
        "test_r2",
        "test_mse",
        "test_mape"
    ]

    delta_df = df.copy()

    for metric in delta_metrics:
        delta_df[f"{metric}_delta"] = (
            delta_df[metric] - baseline[metric]
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    titles = {
        "test_mae_delta": "Δ Test MAE",
        "test_r2_delta": "Δ Test R²",
        "test_mse_delta": "Δ Test MSE",
        "test_mape_delta": "Δ Test MAPE"
    }

    for ax, (metric, title) in zip(axes.flat, titles.items()):

        values = delta_df[metric]

        bars = ax.bar(labels, values)

        ax.axhline(0, linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Diferença vs. baseline")
        ax.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9
            )

    plt.suptitle("Variação das métricas em relação ao baseline", fontsize=15)
    plt.tight_layout()
    plt.show()


    # ============================================================
    # 8. HEATMAP — TESTE
    # ============================================================

    heatmap_df = df.set_index("dataset")[
        ["test_mae", "test_r2", "test_mse", "test_mape"]
    ].copy()

    heatmap_df.index = labels

    plt.figure(figsize=(9, 5))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cmap="coolwarm"
    )

    plt.title("Heatmap das métricas de teste")
    plt.xlabel("Métrica")
    plt.ylabel("Dataset")

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 9. PRIVACY BUDGET × PERFORMANCE
    # ============================================================

    privacy_df = df.copy()

    privacy_df["epsilon"] = [
        np.nan,
        0.1,
        0.5,
        1.0,
        2.0
    ]

    privacy_df = privacy_df.dropna(subset=["epsilon"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    metrics = {
        "test_mae": "Test MAE",
        "test_r2": "Test R²",
        "test_mse": "Test MSE",
        "test_mape": "Test MAPE"
    }

    for ax, (metric, title) in zip(axes.flat, metrics.items()):

        ax.plot(
            privacy_df["epsilon"],
            privacy_df[metric],
            marker="o"
        )

        ax.set_title(title)
        ax.set_xlabel("Privacy budget (ε)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)

    plt.suptitle(
        "Relação entre ε e desempenho do modelo",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()