from attacks import extract_attack_features
from core.experiment_result import ExperimentResult
from .model_runner import run_model
from .attack_runner import run_attacks


def run_machine_learning_experiments(
    model_runner,
    model_name,
    datasets,
    dataset_names,
    seeds,
    test_sizes,
):

    """
    Protocolo experimental padrão do projeto.

    Este método:
    - Chama os métodos responsáveis pelo experimento da aplicação
    - organiza os resultados brutos por execução

    NÃO:
    - altera datasets
    - aplica DP
    - agrega métricas
    """

    experiment_results = []

    for name, df in zip(dataset_names, datasets):

        for seed in seeds:
            for test_size in test_sizes:

                model_metrics_values = run_model(
                    df=df,
                    model_runner=lambda **kwargs: model_runner(
                        **kwargs,
                        seed=seed,
                        test_size=test_size
                    )
                )

                attack_features = extract_attack_features(model_metrics_values)
                attack_metrics_values = run_attacks(attack_features=attack_features)

                experiment_results.append(
                    ExperimentResult(
                        utility_metrics=model_metrics_values,
                        attack_metrics=attack_metrics_values,
                        metadata={
                            "model_name": model_name,
                            "dataset": name,
                            "seed": seed,
                            "test_size": test_size,
                        },
                    )
                )

    return experiment_results
