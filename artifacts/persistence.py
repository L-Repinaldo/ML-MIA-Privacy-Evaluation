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

    artifact_dir = _artifact_dir("evaluation", experiment_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    task_dir = artifact_dir / metadata["tasks"]["task_type"]
    task_dir.mkdir(parents=True, exist_ok=True)

    utility_metrics.to_csv( task_dir / "utility_metrics.csv", index=False,)

    input_leakage.to_pickle( task_dir / "leakage_input.pkl" )

    with open( task_dir / "metadata.json", "w", encoding="utf-8", ) as metadata_file:
        json.dump( metadata, metadata_file, ensure_ascii=False, indent=2,)

    return artifact_dir


def _load_metadata(artifact_path):
    with open(artifact_path / "metadata.json", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

    return metadata


def load_utility_leakage_input(*artifact_path):
    artifact_path = Path(*artifact_path)

    try:
        for task in ("classification", "regression"):
            task_path = artifact_path / task

            if task_path.exists():
                metadata = _load_metadata(task_path)

                leakage_input = pd.read_pickle(
                    task_path / "leakage_input.pkl"
                )

                return leakage_input, metadata

        raise FileNotFoundError(
            f"Nenhuma pasta 'classification' ou 'regression' encontrada em {artifact_path}"
        )

    except Exception:
        raise
        

def load_leakage_artifact(artifact_path):
    artifact_path= Path(artifact_path / "membership_attack")

    metadata= _load_metadata(artifact_path)

    attack_data= pd.read_csv(artifact_path / "attack_metrics.csv")

    return attack_data, metadata




def load_utility_artifact(*artifact_path):

    artifact_path = Path(*artifact_path)

    try:
        for task in ("classification", "regression"):
            task_path = artifact_path / task

            if task_path.exists():
                metadata = _load_metadata(task_path)

                utility_data = pd.read_csv(
                    task_path / "utility_metrics.csv"
                )

                return utility_data, metadata

        raise FileNotFoundError(
            f"Nenhuma pasta 'classification' ou 'regression' encontrada em {artifact_path}"
        )

    except Exception:
        raise



def persist_membership_attack_artifact(
    dir_path,
    metadata,
    attack_metrics,
    attack_results):

    artifact_dir = dir_path / "membership_attack"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    attack_metrics.to_csv(artifact_dir / "attack_metrics.csv", index=False)

    with open(artifact_dir / "attack_results.pkl", "wb") as results_file:
        pickle.dump(attack_results, results_file)

    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    return artifact_dir
