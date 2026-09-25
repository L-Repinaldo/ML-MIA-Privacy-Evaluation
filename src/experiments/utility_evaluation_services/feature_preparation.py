import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing.preprocessor import build_preprocessor
from src.core.prepared_features_config import PreparedFeatures

import pandas as pd

FEATURE_MAPPINGS = {
    "Q001": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7},
    "Q002": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7},
    "Q003": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q004": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},

    "Q007": {
        "A": 0, "B": 1, "C": 2, "D": 3,
        "E": 4, "F": 5, "G": 6, "H": 7,
        "I": 8, "J": 9, "K": 10, "L": 11,
        "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16,
    },

    # Binárias
    "Q008": {"A": 0, "B": 1},
    "Q009": {"A": 0, "B": 1},
    "Q010": {"A": 0, "B": 1},
    "Q011": {"A": 0, "B": 1},
    "Q012": {"A": 0, "B": 1},
    "Q013": {"A": 0, "B": 1},
    "Q014": {"A": 0, "B": 1},
    "Q015": {"A": 0, "B": 1},
    "Q016": {"A": 0, "B": 1},
    "Q017": {"A": 0, "B": 1},
    "Q019": {"A": 0, "B": 1},
    "Q020": {"A": 0, "B": 1},

    "Q021": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
    "Q022": {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
}


def prepare_features(
    name,
    df,
    df_clean,
    task_config,
    split_plan,
    preprocessing_config,
):
    X = df.drop(columns=[task_config.target])
    y = df[task_config.target]

    del df

    X = encode_features(
        df=X,
        mappings=FEATURE_MAPPINGS,
        columns=(
            preprocessing_config.ordinal_columns
            + preprocessing_config.numerical_columns
        ),
    )

    preprocessor = build_preprocessor(
        df=X,
        preprocessing_config=preprocessing_config,
    )

    stratify = y if task_config.task_type == "classification" else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify,
    )

    stratify_temp = (
        y_temp
        if task_config.task_type == "classification"
        else None
    )

    X_validation, _, y_validation, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify_temp,
    )

    # --- DADOS LIMPOS (Apenas para extrair o Teste correto) ---
    X_c = df_clean.drop(columns=[task_config.target])
    y_c = df_clean[task_config.target]

    X_c = encode_features(
        df=X_c,
        mappings=FEATURE_MAPPINGS,
        columns=(
            preprocessing_config.ordinal_columns
            + preprocessing_config.numerical_columns
        ),
    )

    stratify_c = y_c if task_config.task_type == "classification" else None

    # Primeiro split idêntico no dado limpo
    _, X_temp_c, _, y_temp_c = train_test_split(
        X_c,
        y_c,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify_c
    )

    stratify_temp_c = (
        y_temp_c
        if task_config.task_type == "classification"
        else None
    )

    _, X_test_clean, _, y_test_clean = train_test_split(
        X_temp_c,
        y_temp_c,
        test_size=split_plan.test_size,
        random_state=split_plan.seed,
        shuffle=True,
        stratify=stratify_temp_c
    )

    # Limpeza de memória
    del X, y, X_temp, y_temp, X_c, y_c, X_temp_c, y_temp_c
    gc.collect()

    # O preprocessor APRENDE os padrões nos dados de treino (privatizados)
    preprocessor.fit(X_train)

    # Aplicamos a transformação nos três conjuntos
    X_train = preprocessor.transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test_clean = preprocessor.transform(X_test_clean) # <--- Teste limpo transformado

    target_encoder = None

    if task_config.task_type == "classification":
        target_encoder = LabelEncoder()

        y_train = target_encoder.fit_transform(y_train)
        y_validation = target_encoder.transform(y_validation)
        y_test = target_encoder.transform(y_test_clean)

    return PreparedFeatures(
        name=name,
        target=task_config.target,
        task_type=task_config.task_type,
        X_train=X_train,
        X_test=X_test_clean,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        target_encoder=target_encoder,
    )




def encode_features(
    df: pd.DataFrame,
    mappings: dict[str, dict],
    columns: list[str],
) -> pd.DataFrame:

    df = df.copy()

    for column in columns:
        if column not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        mapping = mappings.get(column)

        if mapping is None:
            raise ValueError(
                f"Nenhum mapeamento definido para a coluna '{column}'."
            )

        df[column] = df[column].map(mapping)

    return df
