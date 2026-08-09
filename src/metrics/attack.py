from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def compute_attack_metrics(y_true, y_pred):

    attack_f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    attack_precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    attack_recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
 
    tn, fp, fn, tp = confusion_matrix(y_true= y_true, y_pred= y_pred).ravel().tolist()

    member_acc =  ( tp  / (tp + fn) )
    non_member_acc = ( tn / (tn + fp) )

    attack_acc = accuracy_score(y_true= y_true, y_pred= y_pred)
    advantage = member_acc - (1 - non_member_acc)

    return {
        "attack_acc": attack_acc,
        "attack_f1": attack_f1,
        "attack_precision": attack_precision,
        "attack_recall": attack_recall,
        "member_acc": member_acc,
        "non_member_acc": non_member_acc,
        "advantage": advantage
    }