def validate_experiment_config(experiment_config):

    if not experiment_config.target:
        raise ValueError(
            "A configuração do experimento deve informar um target."
        )

    if experiment_config.task_type not in {
        "regression",
        "classification",
    }:
        raise ValueError(
            "task_type deve ser 'regression' ou 'classification'."
        )

    if experiment_config.task_type == "classification":
        raise NotImplementedError(
            "O pipeline atual suporta apenas regressão."
        )