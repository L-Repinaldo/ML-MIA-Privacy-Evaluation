
from sklearn.metrics import (
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    )

from src.core.results_config import UtilityClassificationResult, UtilityRegressionResult


def compute_utility_metrics(prediction_result, task_type="regression"):

    train_true= prediction_result.y_train_true
    train_pred= prediction_result.y_train_pred

    validation_true= prediction_result.y_validation_true
    validation_pred= prediction_result.y_validation_pred

    test_true= prediction_result.y_test_true
    test_pred= prediction_result.y_test_pred

    if task_type == "regression":

    
        train_mae = mean_absolute_error(y_true= train_true, y_pred= train_pred)
        validation_mae= mean_absolute_error(y_true= validation_true, y_pred= validation_pred)
        test_mae = mean_absolute_error(y_true= test_true, y_pred= test_pred)
    
        generalization_gap = ((train_mae - test_mae) / train_mae ) * 100


        train_r2 = r2_score(y_true= train_true,y_pred= train_pred)
        validation_r2 = r2_score(y_true= validation_true, y_pred= validation_pred)
        test_r2 = r2_score(y_true=  test_true, y_pred=test_pred)

        train_mse= mean_squared_error(y_true= train_true, y_pred= train_pred)
        validation_mse= mean_squared_error(y_true= validation_true, y_pred= validation_pred)
        test_mse= mean_squared_error(y_true= test_true, y_pred= test_pred)

        train_mape= mean_absolute_percentage_error(y_true= train_true, y_pred= train_pred)
        validation_mape= mean_absolute_percentage_error(y_true= validation_true, y_pred= validation_pred)
        test_mape= mean_absolute_percentage_error(y_true= test_true, y_pred= test_pred)

    
        return UtilityRegressionResult(
            task_type= task_type,

            train_mae= train_mae,
            validation_mae= validation_mae,
            test_mae= test_mae,

            train_r2= train_r2,
            validation_r2= validation_r2,
            test_r2= test_r2,

            train_mse= train_mse,
            validation_mse= validation_mse,
            test_mse= test_mse,

            train_mape= train_mape,
            validation_mape= validation_mape,
            test_mape= test_mape,

            generalization_gap= generalization_gap,


        )

    elif task_type == "classification":
        
        train_acc = accuracy_score(y_true= train_true, y_pred= train_pred)
        validation_acc= accuracy_score(y_true= validation_true, y_pred= validation_pred)
        test_acc = accuracy_score(y_true= test_true, y_pred= test_pred)

        train_precision= precision_score(y_true= train_true, y_pred= train_pred, average= 'weighted', zero_division=0)
        validation_precision= precision_score(y_true=validation_true, y_pred=validation_pred, average='weighted', zero_division=0)
        test_precision = precision_score(y_true= test_true, y_pred=test_pred, average='weighted', zero_division=0)

        train_recall= recall_score(y_true= train_true, y_pred=train_pred, average='weighted', zero_division=0)
        validation_recall= recall_score(y_true= validation_true, y_pred=validation_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_true=test_true, y_pred=test_pred, average='weighted', zero_division=0)


        train_f1= f1_score(y_true=train_true, y_pred=train_pred, average='macro', zero_division=0)
        validation_f1= f1_score(y_true= validation_true, y_pred= validation_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_true=test_true, y_pred=test_pred, average='macro', zero_division=0)


        generalization_gap = (train_acc - test_acc) * 100


        return UtilityClassificationResult(

            task_type= task_type,

            train_acc= train_acc,
            validation_acc= validation_acc,
            test_acc= test_acc,

            train_precision= train_precision,
            validation_precision= validation_precision,
            test_precision= test_precision,

            train_recall= train_recall,
            validation_recall= validation_recall,
            test_recall= test_recall,

            train_f1= train_f1,
            validation_f1= validation_f1,
            test_f1= test_f1,

            generalization_gap=  generalization_gap,
        )
    
    else: raise ValueError(f"Unknown task type: {task_type}")