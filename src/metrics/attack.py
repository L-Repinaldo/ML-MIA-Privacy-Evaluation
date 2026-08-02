from sklearn.metrics import confusion_matrix, accuracy_score


def compute_attack_metrics(y_true, y_pred):
    """
    Calcula métricas básicas da eficacia das classificações do ataque.
    """ 

    tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred).ravel().tolist()

    member_acc =  ( tp  / (tp + fn) )
    non_member_acc = ( tn / (tn + fp) )

    attack_acc = accuracy_score(y_true= y_true, y_pred= y_pred)
    advantage = member_acc - (1 - non_member_acc)

    return {
        "attack_acc": attack_acc,
        "member_acc": member_acc,
        "non_member_acc": non_member_acc,
        "advantage": advantage
    }