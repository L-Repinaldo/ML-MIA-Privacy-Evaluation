
import gc

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.core.dataset_preparation import PreparedDataset
from src.metrics import compute_attack_metrics

def _feature_names_classification(n_classes):
    return (
        [f"p_class_{c}" for c in range(n_classes)]
        + ["confidence", "entropy", "true_class_probability", "cross_entropy_loss"]
    )


def _feature_names_regression():
    return ["prediction", "absolute_error", "squared_error"]


def build_classification_features(proba, y_true_encoded):
   
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2:
        raise ValueError("proba deve ser uma matriz 2D (n, n_classes).")

    eps = 1e-12
    n = proba.shape[0]
    n_classes = proba.shape[1]

    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.maximum(row_sums, 1e-12)

    cols = [proba[:, c] for c in range(n_classes)]

    confidence = proba.max(axis=1)
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)

    y_idx = np.asarray(y_true_encoded, dtype=int).ravel()
    if len(y_idx) != n:
        raise ValueError(
            "y_true_encoded deve ter o mesmo número de amostras que proba."
        )
    true_class_probability = proba[np.arange(n), y_idx]
    cross_entropy_loss = -np.log(true_class_probability + eps)

    cols += [confidence, entropy, true_class_probability, cross_entropy_loss]

    return np.column_stack(cols)


def build_regression_features(y_true, y_pred):
  
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho.")

    absolute_error = np.abs(y_true - y_pred)
    squared_error = (y_true - y_pred) ** 2

    return np.column_stack([y_pred, absolute_error, squared_error])


def build_membership_features(
    *,
    task_type,
    proba=None,
    y_true_encoded=None,
    y_true=None,
    y_pred=None,
):
    
    if task_type == "classification":
        if proba is None or y_true_encoded is None:
            raise ValueError(
                "classificação exige `proba` e `y_true_encoded`."
            )
        return build_classification_features(proba, y_true_encoded)

    if task_type == "regression":
        if y_true is None or y_pred is None:
            raise ValueError("regressão exige `y_true` e `y_pred`.")
        return build_regression_features(y_true, y_pred)

    raise ValueError(f"Unknown task type: {task_type}")


def feature_names_for_task(task_type, n_classes=None):

    if task_type == "classification":
        return _feature_names_classification(n_classes)
    if task_type == "regression":
        return _feature_names_regression()
    raise ValueError(f"Unknown task type: {task_type}")




def _partition_indices(n, n_partitions, seed):
    
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    return np.array_split(shuffled, n_partitions)


def _split_member_nonmember(indices, member_fraction, seed, stratify=None):
    
    if member_fraction <= 0.0 or member_fraction >= 1.0:
        raise ValueError("member_fraction deve estar em (0, 1).")

    try:
        member_idx, nonmember_idx = train_test_split(
            indices,
            train_size=member_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        member_idx, nonmember_idx = train_test_split(
            indices,
            train_size=member_fraction,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return member_idx, nonmember_idx


def _subsample_balanced(member_features, nonmember_features, rng):
   
    n = min(len(member_features), len(nonmember_features))
    if n <= 0:
        raise ValueError(
            "Pelo menos um membro e um não-membro são necessários para atacar."
        )

    member_sampled = rng.choice(len(member_features), size=n, replace=False)
    nonmember_sampled = rng.choice(len(nonmember_features), size=n, replace=False)

    X = np.vstack([member_features[member_sampled], nonmember_features[nonmember_sampled]])
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return X, y


def _index_rows(X, indices):
   
    return X[indices]






def _build_attack_model(attack_model_config):
    
    random_state = attack_model_config.get("random_state", 42)

    return XGBClassifier(
        n_estimators=attack_model_config.get("n_estimators", 300),
        max_depth=attack_model_config.get("max_depth", 4),
        learning_rate=attack_model_config.get("learning_rate", 0.1),
        subsample=attack_model_config.get("subsample", 1.0),
        colsample_bytree=attack_model_config.get("colsample_bytree", 1.0),
        min_child_weight=attack_model_config.get("min_child_weight", 1),
        gamma=attack_model_config.get("gamma", 0.0),
        reg_alpha=attack_model_config.get("reg_alpha", 0.0),
        reg_lambda=attack_model_config.get("reg_lambda", 1.0),
        tree_method=attack_model_config.get("tree_method", "hist"),
        n_jobs=attack_model_config.get("n_jobs", 4),
        random_state=random_state,
        verbosity=attack_model_config.get("verbosity", 0),
    )


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
    
    if shadow_config is None or not shadow_config.enabled:
        return None

    n_shadow_models = int(shadow_config.n_shadow_models)
    member_fraction = float(shadow_config.member_fraction)
    attack_test_size = float(shadow_config.attack_test_size)
    attack_model_config = dict(shadow_config.attack_model or {})

    n_pool = X_pool.shape[0]

    partitions = _partition_indices(n_pool, n_shadow_models, seed)

    member_feature_list = []
    nonmember_feature_list = []
    n_classes = None

    for shadow_index, shadow_indices in enumerate(partitions):

        shadow_seed = seed + shadow_index

        stratify = None
        if task_type == "classification":
            stratify = np.asarray(y_pool)[shadow_indices]

        member_idx, nonmember_idx = _split_member_nonmember(
            shadow_indices,
            member_fraction,
            shadow_seed,
            stratify=stratify,
        )

        shadow_data = PreparedDataset(
            name=f"shadow_{shadow_index}",
            X_train=_index_rows(X_pool, member_idx),
            X_test=_index_rows(X_pool, nonmember_idx),
            y_train=np.asarray(y_pool)[member_idx],
            y_test=np.asarray(y_pool)[nonmember_idx],
            target_encoder=None,
        )

        shadow_prediction = model_runner(
            prepared_dataset=shadow_data,
            task_type=task_type,
            seed=shadow_seed,
        )

        member_features = _extract_shadow_features(
            shadow_prediction, task_type, "member"
        )
        nonmember_features = _extract_shadow_features(
            shadow_prediction, task_type, "nonmember"
        )

        if n_classes is None and task_type == "classification":
            n_classes = shadow_prediction.train_proba.shape[1]

        member_feature_list.append(member_features)
        nonmember_feature_list.append(nonmember_features)

        del shadow_prediction, shadow_data, member_features, nonmember_features
        gc.collect()

    rng = np.random.default_rng(seed + n_shadow_models)

    member_all = np.vstack(member_feature_list)
    nonmember_all = np.vstack(nonmember_feature_list)

    X_shadow, y_shadow = _subsample_balanced(member_all, nonmember_all, rng)

    del member_all, nonmember_all, member_feature_list, nonmember_feature_list
    gc.collect()

    X_attack_train, X_attack_val, y_attack_train, y_attack_val = (
        train_test_split(
            X_shadow,
            y_shadow,
            test_size=attack_test_size,
            random_state=seed + n_shadow_models + 1,
            stratify=y_shadow,
        )
    )

    attack_model = _build_attack_model(attack_model_config)
    attack_model.fit(X_attack_train, y_attack_train)

    shadow_val_pred = attack_model.predict(X_attack_val)

    del X_attack_train, y_attack_train
    gc.collect()

    target_member_features = _extract_target_features(
        target_prediction_result, task_type, "member"
    )
    target_nonmember_features = _extract_target_features(
        target_prediction_result, task_type, "nonmember"
    )

    target_rng = np.random.default_rng(seed + 999)
    X_target, y_target = _subsample_balanced(
        target_member_features, target_nonmember_features, target_rng
    )

    target_pred = attack_model.predict(X_target)

    del X_target, target_member_features, target_nonmember_features
    gc.collect()

    metrics = compute_attack_metrics(
        y_true=y_target,
        y_pred=target_pred,
    )

    metrics["attack_type"] = "shadow_model"
    metrics["n_shadow_models"] = n_shadow_models
    metrics["shadow_seed"] = seed
    metrics["attack_features"] = "_".join(
        feature_names_for_task(task_type, n_classes)
    )
    metrics["shadow_val_acc"] = float(
        (shadow_val_pred == y_attack_val).mean()
    )

    return metrics



def _extract_shadow_features(shadow_prediction, task_type, kind):
    
    if task_type == "classification":
        if kind == "member":
            return build_classification_features(
                shadow_prediction.train_proba,
                shadow_prediction.y_train_encoded,
            )
        return build_classification_features(
            shadow_prediction.test_proba,
            shadow_prediction.y_test_encoded,
        )

    if task_type == "regression":
        if kind == "member":
            return build_regression_features(
                shadow_prediction.y_train_true,
                shadow_prediction.y_train_pred,
            )
        return build_regression_features(
            shadow_prediction.y_test_true,
            shadow_prediction.y_test_pred,
        )

    raise ValueError(f"Unknown task type: {task_type}")


def _extract_target_features(target_prediction_result, task_type, kind):

    if task_type == "classification":
        if kind == "member":
            return build_classification_features(
                target_prediction_result.train_proba,
                target_prediction_result.y_train_encoded,
            )
        return build_classification_features(
            target_prediction_result.test_proba,
            target_prediction_result.y_test_encoded,
        )

    if task_type == "regression":
        if kind == "member":
            return build_regression_features(
                target_prediction_result.y_train_true,
                target_prediction_result.y_train_pred,
            )
        return build_regression_features(
            target_prediction_result.y_test_true,
            target_prediction_result.y_test_pred,
        )

    raise ValueError(f"Unknown task type: {task_type}")

