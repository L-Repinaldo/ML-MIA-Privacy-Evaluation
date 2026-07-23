from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(df):
    """
    Preprocessor fixo do experimento.

    - As colunas são derivadas explicitamente do dataset fornecido.
    - O objetivo é manter o mesmo protocolo mesmo com degradação estrutural dos dados.
    """

    target = "salario"

    categorical_cols = [
        col for col in ["cargo", "setor"]
        if col in df.columns
    ]

    numerical_cols = [
        col for col in ["idade", "tempo_na_empresa", "nota_media"]
        if col in df.columns
    ]

    if not categorical_cols and not numerical_cols:
        raise ValueError("Nenhuma feature válida encontrada para o experimento.")

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown='ignore'
                ),
                categorical_cols
            ),
            (
                "numerical",
                "passthrough",
                numerical_cols
            ),
        ],
        remainder="drop"
    )
