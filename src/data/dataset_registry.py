from pathlib import Path

import numpy as np

from .loader import load_data


BASELINE_FILE = "baseline.parquet"
METADATA_FILE = "metadata.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dataset_bundle(
    dataset_name: str,
    dataset_version: str,
    columns: list[str] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
):
    dataset_path = _resolve_dataset_directory(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )

    files = _discover_dataset_files(dataset_path)

    if not files:
        raise FileNotFoundError(
            f"Nenhum dataset encontrado em: {dataset_path}"
        )

    # ---------------------------------------------------------
    # Define os índices da amostra UMA ÚNICA VEZ
    # ---------------------------------------------------------

    sample_indices = None

    if sample_size is not None:
        baseline_name, baseline_file = files[0]

        total_rows = _get_dataset_size(
            baseline_file        )

        sample_size = min(sample_size, total_rows)

        rng = np.random.default_rng(random_state)

        sample_indices = np.sort(
            rng.choice(
                total_rows,
                size=sample_size,
                replace=False,
            )
        )

    # ---------------------------------------------------------
    # Carrega cada dataset separadamente
    # ---------------------------------------------------------

    datasets = []
    dataset_names = []

    for name, file in files:
        df = load_data(
            file,
            columns=columns,
        )

        if sample_indices is not None:
            df = df.iloc[sample_indices]

        df = df.reset_index(drop=True)

        datasets.append(df)
        dataset_names.append(name)

    return {
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
        "datasets": datasets,
        "dataset_names": dataset_names,
    }


def _resolve_dataset_directory(
    dataset_name: str,
    dataset_version: str,
):
    dataset_path = (
        PROJECT_ROOT
        / "src"
        / "data"
        / "datasets"
        / dataset_name
        / dataset_version
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {dataset_path}"
        )

    return dataset_path


def _discover_dataset_files(dataset_path: Path):
    files = []

    baseline = dataset_path / BASELINE_FILE

    if baseline.exists():
        files.append(("baseline", baseline))

    dp_files = sorted(
        dataset_path.glob("dp_eps_*.parquet")
    )

    files.extend(
        (file.stem, file)
        for file in dp_files
    )

    return files


def _get_dataset_size(
    path: Path):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)

    return parquet_file.metadata.num_rows