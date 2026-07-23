import json
from pathlib import Path

import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def list_artifacts(artifacts_dir=ARTIFACTS_DIR):
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        return []

    return sorted(
        [
            path
            for path in artifacts_path.iterdir()
            if path.is_dir()
            and (path / "utility_metrics.csv").exists()
            and (path / "attack_metrics.csv").exists()
            and (path / "metadata.json").exists()
        ],
        key=lambda path: path.name,
    )


def get_latest_artifact(artifacts_dir=ARTIFACTS_DIR):
    artifacts = list_artifacts(artifacts_dir)
    if not artifacts:
        return None

    return artifacts[-1]


def load_artifact(artifact_dir):
    artifact_path = Path(artifact_dir)

    with open(artifact_path / "metadata.json", "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return {
        "path": artifact_path,
        "utility_metrics": pd.read_csv(artifact_path / "utility_metrics.csv"),
        "attack_metrics": pd.read_csv(artifact_path / "attack_metrics.csv"),
        "metadata": metadata,
    }
