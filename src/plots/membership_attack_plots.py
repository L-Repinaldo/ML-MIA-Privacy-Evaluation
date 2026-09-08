import pandas as pd
import matplotlib.pyplot as plt

def plot_membership_attack_results(df):

    # ============================================================
    # CARREGAMENTO
    # ============================================================


    # Remove '*' caso existam no CSV
    df.columns = df.columns.str.replace("*", "", regex=False)


    # ============================================================
    # PREPARAÇÃO
    # ============================================================

    numeric_columns = [
        "n_shadow_models",
        "member_fraction",
        "attack_test_size",
        "shadow_val_acc",
        "attack_acc",
        "attack_f1",
        "attack_precision",
        "attack_recall",
        "member_acc",
        "non_member_acc",
        "advantage",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


    # Ordem dos datasets
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


    dataset_labels = {
        "baseline": "Baseline",
        "dp_eps_0.1": "ε = 0.1",
        "dp_eps_0.5": "ε = 0.5",
        "dp_eps_1.0": "ε = 1.0",
        "dp_eps_2.0": "ε = 2.0",
    }


    # ============================================================
    # 1. DESEMPENHO DO ATAQUE
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["attack_acc"],
        marker="o",
        label="Attack Accuracy",
    )

    plt.plot(
        df["dataset"],
        df["attack_f1"],
        marker="o",
        label="Attack F1",
    )

    plt.plot(
        df["dataset"],
        df["attack_precision"],
        marker="o",
        label="Attack Precision",
    )

    plt.plot(
        df["dataset"],
        df["attack_recall"],
        marker="o",
        label="Attack Recall",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Score")
    plt.xlabel("Dataset")
    plt.title("Desempenho do Membership Inference Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 2. ATTACK ACCURACY
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["attack_acc"],
        marker="o",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Random guessing (50%)",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Attack Accuracy")
    plt.xlabel("Dataset")
    plt.title("Attack Accuracy por nível de privacidade")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 3. MEMBER VS NON-MEMBER ACCURACY
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["member_acc"],
        marker="o",
        label="Member Accuracy",
    )

    plt.plot(
        df["dataset"],
        df["non_member_acc"],
        marker="o",
        label="Non-member Accuracy",
    )

    plt.axhline(
        0.5,
        linestyle="--",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.title("Desempenho do ataque sobre membros e não-membros")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 4. ADVANTAGE
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["advantage"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Advantage")
    plt.xlabel("Dataset")
    plt.title("Membership Inference Advantage")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 5. SHADOW MODEL VS ATTACK MODEL
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["shadow_val_acc"],
        marker="o",
        label="Shadow Model Validation Accuracy",
    )

    plt.plot(
        df["dataset"],
        df["attack_acc"],
        marker="o",
        label="Attack Accuracy",
    )

    plt.axhline(
        0.5,
        linestyle="--",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Accuracy")
    plt.xlabel("Dataset")
    plt.title("Shadow Models vs Membership Attack")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 6. ATTACK F1
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["attack_f1"],
        marker="o",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("F1")
    plt.xlabel("Dataset")
    plt.title("Attack F1 por nível de privacidade")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 7. DIFERENÇA MEMBER - NON-MEMBER
    # ============================================================

    df["member_non_member_gap"] = (
        df["member_acc"] - df["non_member_acc"]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["dataset"],
        df["member_non_member_gap"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xticks(
        range(len(df)),
        [dataset_labels[x] for x in df["dataset"]],
    )

    plt.ylabel("Member Accuracy − Non-member Accuracy")
    plt.xlabel("Dataset")
    plt.title("Assimetria do ataque")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ============================================================
    # 8. HEATMAP DAS MÉTRICAS DO ATAQUE
    # ============================================================

    metrics = [
        "shadow_val_acc",
        "attack_acc",
        "attack_f1",
        "attack_precision",
        "attack_recall",
        "member_acc",
        "non_member_acc",
        "advantage",
    ]

    heatmap_df = df[
        ["dataset"] + metrics
    ].set_index("dataset")

    heatmap_df.index = [
        dataset_labels[x]
        for x in heatmap_df.index
    ]

    plt.figure(figsize=(11, 6))

    plt.imshow(
        heatmap_df,
        aspect="auto",
    )

    plt.xticks(
        range(len(metrics)),
        [
            "Shadow Val.",
            "Attack Acc.",
            "Attack F1",
            "Attack Precision",
            "Attack Recall",
            "Member Acc.",
            "Non-member Acc.",
            "Advantage",
        ],
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(heatmap_df)),)

    plt.colorbar(label="Score")

    plt.title("Resumo das métricas do Membership Inference Attack")

    for i in range(len(heatmap_df)):
        for j in range(len(metrics)):
            plt.text(
                j,
                i,
                f"{heatmap_df.iloc[i, j]:.3f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.show()