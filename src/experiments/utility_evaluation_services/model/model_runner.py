from src.core.prediction_result_config import PredictionResult

from src.experiments.utility_evaluation_services.model.build_model import model_factory

from sklearn.base import ClassifierMixin


def execute_model( prepared_features, model_spec ):

    X_train = prepared_features.X_train
    X_validation = prepared_features.X_validation
    X_test = prepared_features.X_test

    y_train = prepared_features.y_train
    y_validation = prepared_features.y_validation
    y_test = prepared_features.y_test

    encoder = prepared_features.target_encoder

    task_type = prepared_features.task_type

    model = model_factory(model_spec, task_type)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_validation_pred= model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    y_train_encoded = y_train.copy()
    y_validation_encoded = y_validation.copy()
    y_test_encoded = y_test.copy()

    if encoder is not None:

        y_train_pred = encoder.inverse_transform(y_train_pred)
        y_validation_pred = encoder.inverse_transform(y_validation_pred)
        y_test_pred = encoder.inverse_transform(y_test_pred)

        y_train = encoder.inverse_transform(y_train)
        y_validation = encoder.inverse_transform(y_validation)
        y_test = encoder.inverse_transform(y_test)


    if task_type == "classification":
        if not isinstance(model, ClassifierMixin):
            raise TypeError(
                f"Model '{model_spec.model_type}' is not a classifier."
            )

        train_proba = model.predict_proba(X_train)
        validation_proba = model.predict_proba(X_validation)
        test_proba = model.predict_proba(X_test)

        result = PredictionResult(

            y_train_true=y_train,
            y_train_pred=y_train_pred,

            y_validation_true= y_validation,
            y_validation_pred= y_validation_pred,

            y_test_true=y_test,
            y_test_pred=y_test_pred,

            train_proba=train_proba,
            validation_proba= validation_proba,
            test_proba=test_proba,

            y_train_encoded=y_train_encoded,
            y_validation_encoded= y_validation_encoded,
            y_test_encoded=y_test_encoded,

            model=model_spec.model_type
        )

    del X_train, X_test, X_validation, y_train, y_test, y_train_pred, y_validation, y_test_pred, y_validation_pred, train_proba, test_proba, validation_proba

    return result
