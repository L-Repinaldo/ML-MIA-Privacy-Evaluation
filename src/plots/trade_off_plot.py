import pandas as pd
import matplotlib.pyplot as plt


def plot_trade_off(utility_df, attack_df):


    # PREPARAÇÃO

    utility_numeric = [
        "test_balanced_acc",
    ]

    attack_numeric = [
        "advantage",
    ]

    for column in utility_numeric:
        utility_df[column] = pd.to_numeric(
            utility_df[column],
            errors="coerce"
        )

    for column in attack_numeric:
        attack_df[column] = pd.to_numeric(
            attack_df[column],
            errors="coerce"
        )


    # MERGE

    tradeoff_df = utility_df.merge(
        attack_df[
            [
                "dataset",
                "advantage",
            ]
        ],
        on="dataset",
        how="inner"
    )


    # LABELS

    dataset_order = [
                "baseline",
                "dp_eps_0.05",
                "dp_eps_0.1",
                "dp_eps_0.5",
                "dp_eps_1.0",
                "dp_eps_2.0",
                "dp_eps_3.0",
    ]
    

    dataset_labels = {
            "baseline": "Baseline",
            "dp_eps_0.05": "ε = 0.05",
            "dp_eps_0.1": "ε = 0.1",
            "dp_eps_0.5": "ε = 0.5",
            "dp_eps_1.0": "ε = 1.0",
            "dp_eps_2.0": "ε = 2.0",
            "dp_eps_3.0": "ε = 3.0",
    }

    tradeoff_df["dataset"] = pd.Categorical(
        tradeoff_df["dataset"],
        categories=dataset_order,
        ordered=True
    )

    tradeoff_df = tradeoff_df.sort_values("dataset")

    tradeoff_df["label"] = tradeoff_df["dataset"].map(dataset_labels)

    tradeoff_df["advantage_pct"] = tradeoff_df["advantage"] * 100

    tradeoff_df["test_acc_pct"] = tradeoff_df["test_balanced_acc"] * 100


    # GRÁFICO — UTILITY × LEAKAGE

    plt.figure(figsize=(10, 7))

    dp_df = tradeoff_df[
        tradeoff_df["dataset"] != "baseline"
    ]

    plt.scatter(
        dp_df["advantage_pct"],
        dp_df["test_acc_pct"],
        s=100,
        label="DP datasets"
    )

    baseline = tradeoff_df[
        tradeoff_df["dataset"] == "baseline"
    ].iloc[0]

    plt.scatter(
        baseline["advantage_pct"],
        baseline["test_acc_pct"],
        s=180,
        marker="*",
        label="Baseline"
    )



    for _, row in tradeoff_df.iterrows():

        plt.annotate(
            row["label"],
            (
                row["advantage_pct"],
                row["test_acc_pct"]
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=10
        )


    # LINHA DE REFERÊNCIA — RANDOM GUESS

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.text(
        0.02,
        0.02,
        "Menor leakage",
        transform=plt.gca().transAxes,
    )


    # EIXOS

    plt.xlabel("Leakage — Membership Inference Advantage (%)")
    plt.ylabel("Utility — Test Accuracy (%)")

    plt.title(
        "Utility x Leakage Trade-off"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()
    plt.show()