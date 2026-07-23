def compute_utility_metrics(prediction_result):
    """
    Calcula métricas básicas de utilidade do aprendizado.
    """

    train_abs_error = abs(prediction_result.y_train_true - prediction_result.y_train_pred)
    test_abs_error = abs(prediction_result.y_test_true - prediction_result.y_test_pred)

    train_mae = train_abs_error.mean()
    test_mae = test_abs_error.mean()

    generalization_gap = ((test_mae - train_mae) / test_mae ) * 100



    return {
        "train_abs_error": train_abs_error,
        "test_abs_error": test_abs_error,
        "test_mae": test_mae,
        "train_mae": train_mae,
        "generalization_gap_%":  generalization_gap
    }
