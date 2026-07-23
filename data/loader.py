import pandas as pd
from config import ACTIVE_DATASET_DIR


def load_data(path):
    return pd.read_csv(path)


def run():
    def sort_key(path):
        if path.name == "baseline.csv":
            return (0, 0.0)

        epsilon = float(path.stem.replace("dp_eps_", ""))
        return (1, epsilon)

    dataset_files = [ACTIVE_DATASET_DIR / "baseline.csv"]
    dataset_files.extend(ACTIVE_DATASET_DIR.glob("dp_eps_*.csv"))
    dataset_files = [path for path in sorted(dataset_files, key=sort_key) if path.exists()]

    return tuple(load_data(path) for path in dataset_files)
