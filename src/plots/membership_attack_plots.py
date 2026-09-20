import pandas as pd
import matplotlib.pyplot as plt


def plot_membership_attack_results(df):

    # PREPARAÇÃO

    numeric_columns = [
        "n_shadow_models",
        "member_fraction",
        "attack_test_size",
        "shadow_val_acc",
        "attack_acc",
        "attack_f1",
        "attack_precision",
        "member_acc_tpr",
        "non_member_acc_tnr",
        "advantage",
    ]

    for column in numeric_columns:
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

    # 1. ATTACK ACCURACY

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_labels,
        df["attack_acc"],
        marker="o",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Random guessing (50%)",
    )

    plt.ylabel("Attack Accuracy")
    plt.xlabel("Dataset")
    plt.title("Attack Accuracy por nível de privacidade")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. MEMBER VS NON-MEMBER DETECTION

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_labels,
        df["member_acc_tpr"],
        marker="o",
        label="Member Detection",
    )

    plt.plot(
        x_labels,
        df["non_member_acc_tnr"],
        marker="o",
        label="Non-member Detection",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Random guessing (50%)",
    )

    plt.ylabel("Detection Rate")
    plt.xlabel("Dataset")
    plt.title("Member Detection vs Non-member Detection")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. ADVANTAGE

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_labels,
        df["advantage"],
        marker="o",
    )

    plt.axhline(
        0,
        linestyle="--",
        label="No advantage",
    )

    plt.ylabel("Advantage")
    plt.xlabel("Dataset")
    plt.title("Membership Inference Advantage")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 4. SHADOW MODEL VALIDATION ACCURACY

    plt.figure(figsize=(10, 6))

    plt.plot(
        x_labels,
        df["shadow_val_acc"],
        marker="o",
    )

    plt.ylabel("Validation Accuracy")
    plt.xlabel("Dataset")
    plt.title("Shadow Model Validation Accuracy por nível de privacidade")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5. SUMMARY TABLE

    summary_df = df[
        [
            "dataset",
            "shadow_val_acc",
            "attack_acc",
            "attack_f1",
            "attack_precision",
            "member_acc_tpr",
            "non_member_acc_tnr",
            "advantage",
        ]
    ].copy()

    summary_df["dataset"] = summary_df["dataset"].map(
        dataset_labels
    )

    summary_df = summary_df.rename(
        columns={
            "dataset": "Dataset",
            "shadow_val_acc": "Shadow Validation Accuracy",
            "attack_acc": "Attack Accuracy",
            "attack_f1": "Attack F1",
            "attack_precision": "Attack Precision",
            "member_acc_tpr": "Member Detection",
            "non_member_acc_tnr": "Non-member Detection",
            "advantage": "Advantage",
        }
    )

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.axis("off")

    table = ax.table(
        cellText=summary_df.round(3).values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    ax.set_title(
        "Resumo das métricas do ataque de inferência de membresia — Black Box Shadow",
        pad=20,
    )

    plt.tight_layout()
    plt.show()