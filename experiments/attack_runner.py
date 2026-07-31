from metrics import compute_attack_metrics
from attacks import run_membership_inference_attack


def run_attack(attack_features):
    return run_membership_inference_attack(target_outputs=attack_features)


def evaluate_attack(attack_output):
    return compute_attack_metrics(
        y_true=attack_output["y_test"],
        y_pred=attack_output["y_pred"],
    )


def run_attacks(attack_features):
    """Compatibility wrapper for the previous orchestration API."""
    return evaluate_attack(run_attack(attack_features))
