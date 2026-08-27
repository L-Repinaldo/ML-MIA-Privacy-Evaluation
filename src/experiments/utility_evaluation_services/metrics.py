
from sklearn.metrics import (
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
    )

from src.core.results_config import UtilityClassificationResult, UtilityRegressionResult


def compute_utility_metrics(prediction_result, task_type="regression"):

    if task_type == "regression":

        train_abs_error = abs(prediction_result.y_train_true - prediction_result.y_train_pred)
        test_abs_error = abs(prediction_result.y_test_true - prediction_result.y_test_pred)
    
        train_mae = train_abs_error.mean()
        test_mae = test_abs_error.mean()
    
        generalization_gap = ((train_mae - test_mae) / train_mae ) * 100


        train_r2 = r2_score(prediction_result.y_train_true, prediction_result.y_train_pred)
        validation_r2 = r2_score(prediction_result.y_validation_true, prediction_result.y_validation_pred)
        test_r2 = r2_score(prediction_result.y_test_true, prediction_result.y_test_pred)
    
        return UtilityRegressionResult(
            train_abs_error= train_abs_error,
            test_abs_error= test_abs_error,
            test_mae= test_mae,
            train_mae= train_mae,
            generalization_gap=  generalization_gap,
            test_r2= test_r2,
            train_r2= train_r2,
            validation_r2= validation_r2,
            task_type= task_type
        )

    elif task_type == "classification":
        
        train_acc = accuracy_score(prediction_result.y_train_true, prediction_result.y_train_pred)
        validation_acc= accuracy_score(prediction_result.y_validation_true, prediction_result.y_validation_pred)
        test_acc = accuracy_score(prediction_result.y_test_true, prediction_result.y_test_pred)

        test_precision = precision_score(prediction_result.y_test_true, prediction_result.y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(prediction_result.y_test_true, prediction_result.y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(prediction_result.y_test_true, prediction_result.y_test_pred, average='macro', zero_division=0)


        generalization_gap = (train_acc - test_acc) * 100


        return UtilityClassificationResult(
            train_acc= train_acc,
            validation_acc= validation_acc,
            test_acc= test_acc,
            test_precision= test_precision,
            test_recall= test_recall,
            test_f1= test_f1,
            generalization_gap=  generalization_gap,
            task_type= task_type
        )
    
    else: raise ValueError(f"Unknown task type: {task_type}")