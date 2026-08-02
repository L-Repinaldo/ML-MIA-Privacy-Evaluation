from dataclasses import dataclass
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocessor import build_preprocessor


SUPPORTED_TASK_TYPES = {
    "regression",
    "classification",
}


@dataclass
class PreparedDataset:
    name: str

    X_train: object
    X_test: object

    y_train: object
    y_test: object

    target_encoder: LabelEncoder | None = None


def prepare_dataset(
    *,
    name,
    df,
    target,
    task_type,
    preprocessing_config,
    seed,
    test_size,
):
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Task '{task_type}' não suportada.")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrada.")

    preprocessor = build_preprocessor(
        df=df,
        preprocessing_config=preprocessing_config,
    )

    X = df.drop(columns=[target])
    y = df[target]

    t0 = perf_counter()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    print(f"Split: {perf_counter() - t0:.3f}s")

    t1 = perf_counter()

    preprocessor.fit(X_train)

    print(f"Fit: {perf_counter() - t1:.3f}s")

    t2 = perf_counter()

    X_train = preprocessor.transform(X_train)

    print(f"Transform train: {perf_counter() - t2:.3f}s")

    t3 = perf_counter()

    X_test = preprocessor.transform(X_test)

    print(f"Transform test: {perf_counter() - t3:.3f}s")

    print(f"Total preprocess: {perf_counter() - t0:.3f}s")

    encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )

    print(
        f"Número de features geradas: "
        f"{len(encoder.get_feature_names_out())}"
    )

    target_encoder = None

    if task_type == "classification":
        target_encoder = LabelEncoder()

        y_train = target_encoder.fit_transform(y_train)
        y_test = target_encoder.transform(y_test)

    return PreparedDataset(
        name=name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_encoder=target_encoder,
    )