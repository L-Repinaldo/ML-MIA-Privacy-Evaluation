
from src.core.prediction_result import PredictionResult


def run_supervised_model(
        prepared_dataset,
        *,
        task_type,
        model_factory,
    ):
        
        X_train = prepared_dataset.X_train
        X_test = prepared_dataset.X_test

        y_train = prepared_dataset.y_train
        y_test = prepared_dataset.y_test

        encoder = prepared_dataset.target_encoder

        model = model_factory(task_type=task_type)

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if encoder is not None:
            y_train_pred = encoder.inverse_transform(y_train_pred)
            y_test_pred = encoder.inverse_transform(y_test_pred)

            y_train = encoder.inverse_transform(y_train)
            y_test = encoder.inverse_transform(y_test)

        model.task_type_ = task_type
        model.target_encoder_ = encoder

        return PredictionResult(
            y_train_true=y_train,
            y_train_pred=y_train_pred,
            y_test_true=y_test,
            y_test_pred=y_test_pred,
            model=model,
        )