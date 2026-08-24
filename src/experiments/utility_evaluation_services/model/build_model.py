
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


from src.core.models_spec_config import ModelSpec



def model_factory(spec: ModelSpec, task_type: str ):

    if task_type == "classification":

        if spec.model_type == "xgboost_classifier":
                return XGBClassifier(**spec.parameters)
        
        if spec.model_type == "random_forest_classifier":
            return RandomForestClassifier(**spec.parameters)
        
        if spec.model_type == "logistic_regression":
                    return LogisticRegression(**spec.parameters)

    elif task_type == "regression": 

        if spec.model_type == "xgboost_regressor":
            return XGBRegressor(**spec.parameters)


        if spec.model_type == "random_forest_regressor":
            return RandomForestRegressor(**spec.parameters)


        if spec.model_type == "linear_regression":
                return LinearRegression(**spec.parameters)
        
    

    raise ValueError(
        f"Unsupported model type: {spec.model_type}"
    )