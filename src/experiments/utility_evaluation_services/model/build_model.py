
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from src.core.models_spec_config import ModelSpec



def model_factory(spec: ModelSpec, task_type: str ):

    if task_type == "classification":

        if spec.model_type == "xgboost_classifier":
                return XGBClassifier(**spec.parameters)
        
        if spec.model_type == "random_forest_classifier":
            return RandomForestClassifier(**spec.parameters)
        
        if spec.model_type == "logistic_regression":
                    return LogisticRegression(**spec.parameters)

    raise ValueError(
        f"Unsupported model type: {spec.model_type}"
    )