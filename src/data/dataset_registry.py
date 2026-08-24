import json
from pathlib import Path

from .loader import load_data


BASELINE_FILE = "baseline.csv"
METADATA_FILE = "metadata.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]



def load_dataset_bundle( dataset_name: str, dataset_version: str,):

    dataset_path = _resolve_dataset_directory(
        dataset_name= dataset_name,
        dataset_version= dataset_version,
    )

    datasets = []
    dataset_names = []

    baseline = dataset_path / BASELINE_FILE

    if not baseline.exists():
        raise FileNotFoundError(
            f"Baseline não encontrado: {baseline}"
        )

    datasets.append(load_data(baseline))
    dataset_names.append("baseline")

    for file in _discover_dp_files(dataset_path):

        datasets.append(load_data(file))
        dataset_names.append(file.stem)

    metadata = _load_metadata(dataset_path)

    return {
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
        "metadata": metadata,
        "datasets": datasets,
        "dataset_names": dataset_names,
    }


def _resolve_dataset_directory( dataset_name: str, dataset_version: str,):

    dataset_path = (
        Path(f"{PROJECT_ROOT}/src/data/datasets")
        / dataset_name
        / dataset_version
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {dataset_path}"
        )

    return dataset_path


def _discover_dp_files(dataset_path: Path):

    return sorted(
        dataset_path.glob("dp_eps_*.csv")
    )


def _load_metadata(dataset_path: Path):

    metadata_file = dataset_path / METADATA_FILE

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata não encontrado: {metadata_file}"
        )

    with open(metadata_file, encoding="utf-8") as f:
        return json.load(f)