from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from core.prediction_result import PredictionResult

def run_linear_regression(df, preprocessor, *, target="salario", test_size=0.3, seed=42):

    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado no dataset.")
    
    X = df.drop(columns=[target])
    y = df[target]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    model.preprocessor_ = preprocessor

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    return PredictionResult(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
        model=model,
    )