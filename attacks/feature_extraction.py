def extract_attack_features(utility_metrics):
    return {
        "train_abs_error": utility_metrics["train_abs_error"],
        "test_abs_error": utility_metrics["test_abs_error"],
    }
