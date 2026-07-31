from xgboost import XGBClassifier, XGBRegressor

from .common import run_supervised_model



def run_xgboost(
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
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": seed,
        "verbosity": 0,
    }

    if task_type == "regression":
        return XGBRegressor(
            objective="reg:squarederror",
            **common_params,
        )

    return XGBClassifier(**common_params)