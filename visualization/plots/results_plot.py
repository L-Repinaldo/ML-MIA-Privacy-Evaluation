import matplotlib.pyplot as plt
import pandas as pd

def _get_by_model(results, model_name): 
    df = results[results["model"] == model_name]
    if df.empty:
        raise ValueError(f"Modelo {model_name} não encontrado nos resultados.")
    return df

def plot_tables_chart(results: pd.DataFrame, title: str):
    """
    Plota tabelas de métricas separadas por modelo, empilhadas verticalmente.
    
    results: DataFrame com coluna 'model' e métricas
    title_prefix: string que será usada no título de cada tabela
    """
    models = results['model'].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(10, 3 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, model in zip(axes, models):
        df_model = _get_by_model(results, model)
        table_df = df_model.drop(columns=['model'])
        
        ax.axis('off')
        tbl = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        
        ax.set_title(f"{title}: {model}", fontweight='bold')
    
    plt.tight_layout()
    plt.show()