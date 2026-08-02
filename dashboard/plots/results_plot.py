
from common import get_by_model

import matplotlib.pyplot as plt
import pandas as pd


def plot_tables_chart(results: pd.DataFrame, title: str):
    
    models = results['model'].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(10, 3 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, model in zip(axes, models):
        df_model = get_by_model(results, model)
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


def plot_attack_metrics(df_attack):
    plot_tables_chart(results=df_attack, title="Metricas ataque ")

def plot_utility_metrics(df_utility):
    plot_tables_chart(results=df_utility, title="Metricas utilidade ")

def plot_generalization_gap(df_generalization):
    plot_tables_chart(df_generalization, title= "Gap Generalização")
