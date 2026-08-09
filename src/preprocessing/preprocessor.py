
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(df, preprocessing_config):

    preprocessing = preprocessing_config

    categorical_columns = [
        column
        for column in preprocessing.categorical_columns
        if column in df.columns
    ]

    numerical_columns = [
        column
        for column in preprocessing.numerical_columns
        if column in df.columns
    ]

    if not categorical_columns and not numerical_columns:
        raise ValueError(
            "Nenhuma coluna configurada para preprocessamento foi encontrada."
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing.impute_categorical,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    drop=preprocessing.one_hot_drop,
                    handle_unknown=preprocessing.handle_unknown,
                    sparse_output=True,  
                ),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing.impute_numeric,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,  
    )

    return preprocessor