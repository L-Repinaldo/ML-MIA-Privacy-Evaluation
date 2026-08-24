import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocessor import build_preprocessor
from src.core.prepared_features_config import PreparedFeatures


def prepare_features(
    name,
    df,
    task_config,
    split_plan,
    preprocessing_config,
):
    X = df.drop(columns=[task_config.target])
    y = df[task_config.target]

    del df

    preprocessor = build_preprocessor(
        df=X,
        preprocessing_config=preprocessing_config,
    )

    stratify = y if task_config.task_type == "classification" else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify,
    )

    stratify_temp = (
        y_temp
        if task_config.task_type == "classification"
        else None
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify_temp,
    )

    del X, y, X_temp, y_temp
    gc.collect()

    preprocessor.fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test = preprocessor.transform(X_test)

    target_encoder = None

    if task_config.task_type == "classification":
        target_encoder = LabelEncoder()

        y_train = target_encoder.fit_transform(y_train)
        y_validation = target_encoder.transform(y_validation)
        y_test = target_encoder.transform(y_test)

    return PreparedFeatures(
        name=name,
        target=task_config.target,
        task_type=task_config.task_type,
        X_train=X_train,
        X_test=X_test,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        target_encoder=target_encoder,
    )