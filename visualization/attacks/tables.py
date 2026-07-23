from visualization.plots.results_plot import plot_tables_chart


def plot_attack_metrics(df_attack):
    plot_tables_chart(results=df_attack, title="Metricas ataque ")
