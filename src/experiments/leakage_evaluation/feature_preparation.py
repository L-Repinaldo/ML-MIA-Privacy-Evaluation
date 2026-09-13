import numpy as np



def build_membership_features(
    *,
    task_type: str,
    proba=None,
    y_true_encoded=None,
) -> np.ndarray:

    if task_type == "classification":
        if proba is None or y_true_encoded is None:
            raise ValueError("Classificação exige proba e y_true_encoded.")

        return build_classification_features( proba, y_true_encoded)

    raise ValueError(
        f"Unknown task type: {task_type}"
    )


def build_classification_features( proba, y_true_encoded,) -> np.ndarray:

    proba = np.asarray(proba, dtype=np.float64)

    if proba.ndim != 2:
        raise ValueError("proba deve ser uma matriz 2D (n_samples, n_classes).")

    n_samples, n_classes = proba.shape

    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.maximum(row_sums, 1e-12)

    y_true_encoded = np.asarray(
        y_true_encoded,
        dtype=int,
    ).ravel()

    if len(y_true_encoded) != n_samples:
        raise ValueError( "y_true_encoded deve ter o mesmo número de amostras que proba." )

    confidence = proba.max(axis=1)

    entropy = -np.sum(
        proba * np.log(proba + 1e-12),
        axis=1,
    )

    true_class_probability = proba[
        np.arange(n_samples),
        y_true_encoded,
    ]

    cross_entropy_loss = -np.log(
        true_class_probability + 1e-12
    )

    return np.column_stack(
        [
            proba,
            confidence,
            entropy,
            true_class_probability,
            cross_entropy_loss,
        ]
    )


def feature_names_for_task( task_type: str, n_classes: int | None = None) -> list[str]:

    if task_type == "classification":
        if n_classes is None:
            raise ValueError( "n_classes é obrigatório para classificação.")

        return _classification_feature_names(n_classes)

    raise ValueError(f"Unknown task type: {task_type}")



def _classification_feature_names(n_classes: int) -> list[str]:
    return [
        *[f"p_class_{c}" for c in range(n_classes)],
        "confidence",
        "entropy",
        "true_class_probability",
        "cross_entropy_loss",
    ]
