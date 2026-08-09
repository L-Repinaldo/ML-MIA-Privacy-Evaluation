from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .common import run_supervised_model


def run_linear_or_logistic_regression(
    prepared_dataset,
    *,
    task_type="regression",
    seed=42,
):
    return run_supervised_model(
        prepared_dataset=prepared_dataset,
        task_type=task_type,
        model_factory=lambda **kwargs: _build_model(seed=seed, **kwargs),
    )


def _build_model(*, task_type, seed):
    if task_type == "regression":
        return LinearRegression()

    return make_pipeline(
        StandardScaler(with_mean=False),  
        LogisticRegression(
            max_iter=2000,  
            solver='saga',  
            random_state=seed,
        )
    )
