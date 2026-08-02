from config import get_experiment_config
from src.core.experimental_pipeline import ExperimentalPipeline


def main():

    experiment_config = get_experiment_config()

    pipeline = ExperimentalPipeline(
        experiment_config=experiment_config,
    )

    pipeline.run()


if __name__ == "__main__":
    main()