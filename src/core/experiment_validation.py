def validate_experiment_config(experiment_config):

    if not experiment_config.tasks:
        raise ValueError(
            "A configuração do experimento deve informar pelo menos uma tarefa."
        )

    for task in experiment_config.tasks:
        if not task.target:
            raise ValueError(
                f"Cada tarefa deve informar um target. Tarefa: {task.task_type}"
            )

        if task.task_type not in {
            "regression",
            "classification",
        }:
            raise ValueError(
                f"task_type deve ser 'regression' ou 'classification'. Tarefa: {task.task_type}"
            )

        if not task.active_models:
            raise ValueError(
                f"Cada tarefa deve informar pelo menos um modelo. Tarefa: {task.task_type}"
            )
