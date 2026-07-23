from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
DATASET_VERSION = "v-2026-03-02_18-10-54"

ACTIVE_DATASET_DIR = DATASETS_DIR / DATASET_VERSION

