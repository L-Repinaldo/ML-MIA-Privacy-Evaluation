import numpy as np





def build_membership_features(
    *,
    task_type: str,
    proba=None,
    y_true_encoded=None,
    y_true=None,
    y_pred=None,
) -> np.ndarray:

    if task_type == "classification":
        if proba is None or y_true_encoded is None:
            raise ValueError("Classificação exige proba e y_true_encoded.")

        return build_classification_features( proba, y_true_encoded)

    if task_type == "regression":
        if y_true is None or y_pred is None:
            raise ValueError("Regressão exige y_true e y_pred.")

        return build_regression_features(y_true, y_pred)

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


def build_regression_features(y_true, y_pred) -> np.ndarray:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).ravel()

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError( "y_true e y_pred devem ter o mesmo tamanho.")

    error = y_true - y_pred

    absolute_error = np.abs(error)
    squared_error = error**2

    return np.column_stack(
        [
            y_pred,
            absolute_error,
            squared_error,
        ]
    )


def feature_names_for_task( task_type: str, n_classes: int | None = None) -> list[str]:

    if task_type == "classification":
        if n_classes is None:
            raise ValueError( "n_classes é obrigatório para classificação.")

        return _classification_feature_names(n_classes)

    if task_type == "regression":
        return _regression_feature_names()

    raise ValueError(f"Unknown task type: {task_type}")



def _classification_feature_names(n_classes: int) -> list[str]:
    return [
        *[f"p_class_{c}" for c in range(n_classes)],
        "confidence",
        "entropy",
        "true_class_probability",
        "cross_entropy_loss",
    ]


def _regression_feature_names() -> list[str]:
    return [
        "prediction",
        "absolute_error",
        "squared_error",
    ]
