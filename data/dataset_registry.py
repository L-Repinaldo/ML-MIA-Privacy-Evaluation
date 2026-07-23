import re
from pathlib import Path

from config import DATASETS_DIR
from .loader import load_data


def _dataset_name(path):
    if path.name == "baseline.csv":
        return "baseline"

    return path.stem.replace("dp_", "")


def _dataset_sort_key(path):
    if path.name == "baseline.csv":
        return (0, 0.0)

    match = re.match(r"dp_eps_(.+)\.csv$", path.name)
    if match:
        return (1, float(match.group(1)))

    return (2, path.name)


def discover_dataset_files(dataset_version):
    dataset_dir = Path(DATASETS_DIR) / dataset_version
    dataset_files = [dataset_dir / "baseline.csv"]
    dataset_files.extend(dataset_dir.glob("dp_eps_*.csv"))
    return [path for path in sorted(dataset_files, key=_dataset_sort_key) if path.exists()]


def load_registered_datasets(dataset_version, active_datasets=None):
    dataset_files = discover_dataset_files(dataset_version)
    selected_names = set(active_datasets) if active_datasets is not None else None

    names = []
    datasets = []
    for path in dataset_files:
        name = _dataset_name(path)
        if selected_names is not None and name not in selected_names:
            continue

        names.append(name)
        datasets.append(load_data(path))

    return datasets, names
