def add_epsilon_column(utility_metrics):
    epsilon_map = {
        "baseline": 0.0,
        "dp_eps_0.1": 0.1,
        "dp_eps_0.5": 0.5,
        "dp_eps_1.0": 1.0,
        "dp_eps_2.0": 2.0,
    }

    df = utility_metrics.copy()
    df["epsilon"] = df["dataset"].map(epsilon_map)

    return df