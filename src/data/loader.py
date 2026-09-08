import pandas as pd


def load_data(path, columns=None):
    return pd.read_parquet(
        path,
        columns=columns,
    )