import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MAX_SAMPLES = 100000

def run_membership_inference_attack(target_outputs):

    train_loss = target_outputs['train_loss']
    test_loss  = target_outputs['test_loss']
    train_confidence = target_outputs['train_confidence']
    test_confidence  = target_outputs['test_confidence']

    n = min(len(train_loss), len(test_loss), MAX_SAMPLES)

    train_sample = np.random.choice(train_loss, n, replace=False)
    test_sample  = np.random.choice(test_loss, n, replace=False)

    train_confidence = np.random.choice(train_confidence, n, replace=False) if train_confidence is not None else None
    test_confidence  = np.random.choice(test_confidence, n, replace=False) if test_confidence is not None else None

    train_X = np.column_stack([
        train_sample,
        train_confidence[:n] if train_confidence is not None else np.zeros(n),
    ])

    test_X = np.column_stack([
        test_sample,
        test_confidence[:n] if test_confidence is not None else np.zeros(n),
    ])

    X = np.vstack([train_X, test_X])
    y = np.concatenate([
        np.ones(n),
        np.zeros(n)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    attack_model = train_attack_model(X_train, y_train)

    y_pred = attack_model.predict(X_test)

    return {
        "y_pred": y_pred,
        "y_test": y_test 
    }


def train_attack_model(attack_X, attack_y):
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        )
    clf.fit(attack_X, attack_y)
    return clf