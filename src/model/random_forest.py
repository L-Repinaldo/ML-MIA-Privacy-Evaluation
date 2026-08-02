from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .common import run_supervised_model



def run_random_forest(
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
    common_params = {
        "n_estimators": 100,
        "max_depth": 8,
        "min_samples_leaf": 8,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": seed,
        "n_jobs": 4,
    }

    model_class = (
        RandomForestRegressor
        if task_type == "regression"
        else RandomForestClassifier
    )

    return model_class(**common_params)