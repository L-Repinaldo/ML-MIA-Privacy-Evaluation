from sklearn.linear_model import LinearRegression, LogisticRegression

from .common import run_supervised_model



def run_linear_regression(
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

    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
    )