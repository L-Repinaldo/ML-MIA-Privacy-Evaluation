import json
import pickle
from pathlib import Path

import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parent


def _artifact_dir(experiment_type, experiment_id):
    prefix = f"{experiment_type}_"
    artifact_name = (
        experiment_id
        if str(experiment_id).startswith(prefix)
        else f"{prefix}{experiment_id}"
    )

    return ARTIFACTS_DIR / experiment_type / artifact_name


def persist_utility_artifact(experiment_id, metadata, utility_metrics, input_leakage):


    artifact_dir = _artifact_dir("utility", experiment_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    utility_metrics.to_csv(artifact_dir / "utility_metrics.csv", index=False)

    input_leakage.to_pickle(artifact_dir / "leakage_input.pkl")

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return artifact_dir


def load_utility_artifact(artifact_path):
    artifact_path = Path(artifact_path)

    with open(artifact_path / "metadata.json", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    leakage_input = pd.read_pickle(artifact_path / "leakage_input.pkl")

    return metadata, leakage_input


def persist_membership_attack_artifact(
    experiment_id,
    metadata,
    attack_metrics,
    attack_results,
):
    artifact_dir = _artifact_dir("mia", experiment_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    attack_metrics.to_csv(artifact_dir / "attack_metrics.csv", index=False)

    with open(artifact_dir / "attack_results.pkl", "wb") as results_file:
        pickle.dump(attack_results, results_file)

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return artifact_dir
