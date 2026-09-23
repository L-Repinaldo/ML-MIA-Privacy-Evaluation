
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score, 
    f1_score,
    )

from src.core.results_config import UtilityClassificationResult


def compute_utility_metrics(prediction_result, task_type="classification"):

    train_true= prediction_result.y_train_true
    train_pred= prediction_result.y_train_pred

    validation_true= prediction_result.y_validation_true
    validation_pred= prediction_result.y_validation_pred

    test_true= prediction_result.y_test_true
    test_pred= prediction_result.y_test_pred


    if task_type == "classification":
        
        train_balanced_acc= balanced_accuracy_score(y_true= train_true, y_pred= train_pred)
        validation_balanced_acc= balanced_accuracy_score(y_true= validation_true, y_pred= validation_pred)
        test_balanced_acc = balanced_accuracy_score(y_true= test_true, y_pred= test_pred)

        train_precision= precision_score(y_true= train_true, y_pred= train_pred, average= 'macro', zero_division=0)
        validation_precision= precision_score(y_true=validation_true, y_pred=validation_pred, average='macro', zero_division=0)
        test_precision = precision_score(y_true= test_true, y_pred=test_pred, average='macro', zero_division=0)

        train_f1= f1_score(y_true=train_true, y_pred=train_pred, average='macro', zero_division=0)
        validation_f1= f1_score(y_true= validation_true, y_pred= validation_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_true=test_true, y_pred=test_pred, average='macro', zero_division=0)


        generalization_gap = (train_balanced_acc - test_balanced_acc) * 100


        return UtilityClassificationResult(

            task_type= task_type,

            train_balanced_acc= train_balanced_acc,
            validation_balanced_acc= validation_balanced_acc,
            test_balanced_acc= test_balanced_acc,

            train_precision= train_precision,
            validation_precision= validation_precision,
            test_precision= test_precision,

            train_f1= train_f1,
            validation_f1= validation_f1,
            test_f1= test_f1,
            
            generalization_gap=  generalization_gap,
        )
    
    else: raise ValueError(f"Unknown task type: {task_type}")