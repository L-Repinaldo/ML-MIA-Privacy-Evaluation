import gc

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.experiments.lekage_evaluation.attacks import shadow

from src.core.models_spec_config import ModelSpec


def run_shadow_model_membership_inference_attack(
    *,
    model_runner,
    task_type,
    X_pool,
    y_pool,
    target_prediction_result,
    shadow_config,
    seed,
):

    if (
        shadow_config is None
        or not shadow_config.enabled
    ):
        return None

    n_shadow_models = int(
        shadow_config.n_shadow_models
    )

    member_fraction = float(
        shadow_config.member_fraction
    )

    attack_test_size = float(
        shadow_config.attack_test_size
    )

    attack_model_config = shadow_config.attack_model


    member_features, nonmember_features = (
        shadow.run_shadow_models(
            model_runner=model_runner,
            task_type=task_type,
            X_pool=X_pool,
            y_pool=y_pool,
            n_shadow_models=n_shadow_models,
            member_fraction=member_fraction,
            seed=seed,
        )
    )


    rng = np.random.default_rng(
        seed + n_shadow_models
    )

    X_shadow, y_shadow = shadow._subsample_balanced(
        member_features,
        nonmember_features,
        rng,
    )

    del member_features
    del nonmember_features

    gc.collect()

    (
        X_attack_train,
        X_attack_val,
        y_attack_train,
        y_attack_val,
    ) = train_test_split(
        X_shadow,
        y_shadow,
        test_size=attack_test_size,
        random_state=seed + n_shadow_models + 1,
        stratify=y_shadow,
    )


    attack_model = _build_attack_model(
        attack_model_config
    )

    attack_model.fit(
        X_attack_train,
        y_attack_train,
    )

    shadow_val_pred = attack_model.predict(
        X_attack_val
    )

    del X_attack_train
    del y_attack_train

    gc.collect()


    target_member_features = (
        shadow.extract_prediction_features(
            target_prediction_result,
            task_type,
            member=True,
        )
    )

    target_nonmember_features = (
        shadow.extract_prediction_features(
            target_prediction_result,
            task_type,
            member=False,
        )
    )


    target_rng = np.random.default_rng(
        seed + 999
    )

    X_target, y_target = shadow._subsample_balanced(
        target_member_features,
        target_nonmember_features,
        target_rng,
    )

    target_pred = attack_model.predict(
        X_target
    )

    return {
        "attack_model": attack_model,
        "shadow_validation_predictions": shadow_val_pred,
        "shadow_validation_labels": y_attack_val,
        "target_predictions": target_pred,
        "target_labels": y_target,
    }


def run_membership_inference_attack(**kwargs):
    return run_shadow_model_membership_inference_attack(**kwargs)


def _build_attack_model( config: ModelSpec) -> XGBClassifier:

    if config.model_type != "xgboost_classifier":
        raise ValueError(
            f"Unsupported attack model: {config.model_type}"
        )

    return XGBClassifier(
        **config.parameters
    )