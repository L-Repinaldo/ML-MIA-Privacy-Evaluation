import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def run_membership_inference_attack(target_outputs):

    train_abs_error = target_outputs['train_abs_error']
    test_abs_error  = target_outputs['test_abs_error']

    n = min(len(train_abs_error), len(test_abs_error))

    train_sample = np.random.choice(train_abs_error, n, replace=False)
    test_sample  = np.random.choice(test_abs_error, n, replace=False)

    X = np.concatenate([train_sample, test_sample]).reshape(-1, 1)
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

# -----------------------------
# Train attack model
# -----------------------------
def train_attack_model(attack_X, attack_y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(attack_X, attack_y)
    return clf