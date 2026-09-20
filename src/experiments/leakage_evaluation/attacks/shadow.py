import gc

import numpy as np
from sklearn.model_selection import train_test_split

from src.core.prepared_features_config import PreparedFeatures

from src.experiments.leakage_evaluation.feature_preparation import build_membership_features


def _partition_indices(
    n: int,
    n_partitions: int,
    seed: int,
) -> list[np.ndarray]:

    rng = np.random.default_rng(seed)

    shuffled = rng.permutation(n)

    return np.array_split(
        shuffled,
        n_partitions,
    )


def _split_member_nonmember(
    indices,
    member_fraction: float,
    seed: int,
    stratify=None,
):
    if not 0.0 < member_fraction < 1.0:
        raise ValueError(
            "member_fraction deve estar em (0, 1)."
        )

    try:
        return train_test_split(
            indices,
            train_size=member_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )

    except ValueError:
        return train_test_split(
            indices,
            train_size=member_fraction,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def _subsample_balanced(
    member_features,
    nonmember_features,
    rng,
):
    n = min(
        len(member_features),
        len(nonmember_features),
    )

    if n <= 0:
        raise ValueError(
            "Pelo menos um membro e um não-membro "
            "são necessários para o ataque."
        )

    member_indices = rng.choice(
        len(member_features),
        size=n,
        replace=False,
    )

    nonmember_indices = rng.choice(
        len(nonmember_features),
        size=n,
        replace=False,
    )

    X = np.vstack(
        [
            member_features[member_indices],
            nonmember_features[nonmember_indices],
        ]
    )

    y = np.concatenate(
        [
            np.ones(n, dtype=int),
            np.zeros(n, dtype=int),
        ]
    )

    return X, y


def _index_rows(
    X,
    indices,
):
    if hasattr(X, "iloc"):
        return X.iloc[indices]

    return X[indices]


def _prediction_value(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]

    return getattr(prediction, key)


def extract_prediction_features(
    prediction,
    task_type: str,
    *,
    member: bool,
):
    if task_type == "classification":

        if member:
            return build_membership_features(
                task_type="classification",
                proba=_prediction_value(prediction, "train_proba"),
                y_true_encoded=_prediction_value(prediction, "y_train_encoded"),
            )

        return build_membership_features(
            task_type="classification",
            proba=_prediction_value(prediction, "test_proba"),
            y_true_encoded=_prediction_value(prediction, "y_test_encoded"),
        )


    raise ValueError(
        f"Unknown task type: {task_type}"
    )


def run_shadow_models(
    *,
    model_runner,
    task_type: str,
    X_pool,
    y_pool,
    n_shadow_models: int,
    member_fraction: float,
    seed: int,
):
    partitions = _partition_indices(
        n=X_pool.shape[0],
        n_partitions=n_shadow_models,
        seed=seed,
    )

    member_feature_list = []
    nonmember_feature_list = []

    y_pool = np.asarray(y_pool)

    for shadow_index, shadow_indices in enumerate(partitions):

        shadow_seed = seed + shadow_index

        stratify = None

        if task_type == "classification":
            stratify = y_pool[shadow_indices]

        member_idx, nonmember_idx = (
            _split_member_nonmember(
                shadow_indices,
                member_fraction,
                shadow_seed,
                stratify=stratify,
            )
        )

        shadow_data = PreparedFeatures(
            name=f"shadow_{shadow_index}",
            target="",
            task_type=task_type,
            X_train=_index_rows(
                X_pool,
                member_idx,
            ),
            X_validation=_index_rows(
                X_pool,
                nonmember_idx,
            ),
            X_test=_index_rows(
                X_pool,
                nonmember_idx,
            ),
            y_train=y_pool[member_idx],
            y_validation=y_pool[nonmember_idx],
            y_test=y_pool[nonmember_idx],
            target_encoder=None,
        )

        prediction = model_runner(
            prepared_dataset=shadow_data,
            task_type=task_type,
            seed=shadow_seed,
        )

        member_features = extract_prediction_features(
            prediction,
            task_type,
            member=True,
        )

        nonmember_features = extract_prediction_features(
            prediction,
            task_type,
            member=False,
        )

        member_feature_list.append(
            member_features
        )

        nonmember_feature_list.append(
            nonmember_features
        )

        del prediction
        del shadow_data
        del member_features
        del nonmember_features

        gc.collect()

    member_all = np.vstack(
        member_feature_list
    )

    nonmember_all = np.vstack(
        nonmember_feature_list
    )

    return member_all, nonmember_all