from metrics import compute_attack_metrics
from attacks import run_membership_inference_attack

def run_attacks(attack_features):
    """
    Protocolo experimental padrão do projeto.

    Este método:
    - executa ataque de inferência
    - mede se o índice de vazamentos
    - organiza os resultados

    NÃO:
    - altera datasets
    - aplica DP
    - otimiza modelos
    """
    
    mia_output = run_membership_inference_attack(target_outputs=attack_features)


    return compute_attack_metrics(y_true= mia_output['y_test'], y_pred= mia_output['y_pred'])
