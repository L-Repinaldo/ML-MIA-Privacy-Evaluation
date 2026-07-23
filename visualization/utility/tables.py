from visualization.plots.results_plot import plot_tables_chart


def plot_utility_metrics(df_utility):
    plot_tables_chart(results=df_utility, title="Metricas utilidade ")

def plot_generalization_gap(df_generalization):
    plot_tables_chart(df_generalization, title= "Gap Generalização")
