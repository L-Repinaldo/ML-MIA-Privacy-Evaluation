def extract_attack_features(utility_metrics, task_type="regression"):

    if task_type == "regression":
        return {
            "train_loss": utility_metrics["train_abs_error"],
            "test_loss": utility_metrics["test_abs_error"],
            "train_confidence": None,
            "test_confidence": None
        }
    elif task_type == "classification":
        return {
            "train_loss": utility_metrics["train_loss"],
            "test_loss": utility_metrics["test_loss"],
            "train_confidence": utility_metrics["train_confidence"],
            "test_confidence": utility_metrics["test_confidence"],
        }
    else:
        raise ValueError(f"Unknown task type: {task_type}")