
from xgboost import XGBClassifier

from src.core.models_spec_config import ModelSpec



def model_factory(spec: ModelSpec, task_type: str ):

    if task_type == "classification":

        if spec.model_type == "xgboost_classifier":
                return XGBClassifier(**spec.parameters)

    raise ValueError(
        f"Unsupported model type: {spec.model_type}"
    )