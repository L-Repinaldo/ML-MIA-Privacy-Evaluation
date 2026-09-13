
from src.core.models_spec_config import ModelSpec
from src.experiments.utility_evaluation_services.model import model_runner



def build_shadow_model(model_spec):
    def execute_shadow_model(prepared_dataset, seed, task_type=None):
        parameters = dict(model_spec.parameters)

        if "random_state" in parameters:
            parameters["random_state"] = seed

        seeded_model_spec = ModelSpec(
            name=model_spec.name,
            model_type=model_spec.model_type,
            parameters=parameters,
        )

        return model_runner.execute_model(
            prepared_features=prepared_dataset,
            model_spec=seeded_model_spec,
        )

    return execute_shadow_model