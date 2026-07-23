from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from core.prediction_result import PredictionResult

def run_xgboost(df, preprocessor, *, target="salario", test_size=0.3, seed=42):
    """
    Executa o modelo Gradient Boosted Trees (XGBoost).

    Papel deste modelo no protocolo:
    - Representar modelos de alta capacidade
    - Capturar relações não lineares complexas
    - Avaliar tendência a memorizar padrões locais e ruído

    Responsabilidades:
    - realizar split controlado
    - aplicar o pré-processamento definido no pipeline
    - treinar modelo determinístico (semente fixa)
    - gerar predições de treino e teste

    """

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

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=5.0,
        random_state=seed,
        objective="reg:squarederror",
        verbosity=0
    )

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
