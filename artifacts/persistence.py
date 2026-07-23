import json
from datetime import datetime
from pathlib import Path


ARTIFACTS_DIR = Path(__file__).resolve().parent


def persist_experiment_artifacts(experiment_id, df_utility, df_attack, metadata):
    experiment_dir = ARTIFACTS_DIR / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    df_utility.to_csv(experiment_dir / "utility_metrics.csv", index=False)
    df_attack.to_csv(experiment_dir / "attack_metrics.csv", index=False)

    with open(experiment_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return experiment_dir


def build_artifact_metadata(config, timestamp):
    return {
        "dataset_version": config.dataset_version,
        "timestamp": timestamp.isoformat(),
        "modelos": config.model_names,
        "seeds": list(config.seeds),
        "test_sizes": list(config.test_sizes),
    }


def build_experiment_id(timestamp=None):
    timestamp = timestamp or datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")
