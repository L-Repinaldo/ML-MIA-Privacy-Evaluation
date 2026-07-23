from metrics import compute_utility_metrics
from preprocessing import build_preprocessor


def run_model(df, model_runner):
    """
    Protocolo experimental padrão do projeto.

    Este método:
    - executa o modelo como instrumento de medição
    - mede utilidade
    - mede estabilidade
    - executa ataque de inferência
    - organiza os resultados

    NÃO:
    - altera datasets
    - aplica DP
    - otimiza modelos
    """


    preprocessor = build_preprocessor(df=df)

    run_output = model_runner(
        df=df,
        preprocessor=preprocessor,
    )

    return compute_utility_metrics(prediction_result=run_output)
