import plotly.express as px


def plot_privacy_utility_tradeoff(tradeoff_df):
    
    plot_df = tradeoff_df.dropna(subset=["utility_loss", "advantage"]).copy()
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(
        by=["model", "epsilon"], na_position="last"
    )

    fig = px.line(
        plot_df,
        x="utility_loss",
        y="advantage",
        color="model",
        text="epsilon_label",
        markers=True,
        hover_data={
            "epsilon_label": True,
            "utility_metric_value": ":.3f",
            "baseline_metric_value": ":.3f",
            "utility_loss": ":.3f",
            "attack_acc": ":.3f",
            "member_acc": ":.3f",
            "non_member_acc": ":.3f",
        },
        labels={
            "utility_loss": "Perda relativa de utilidade vs baseline",
            "advantage": "Advantage",
            "model": "Modelo",
        },
        title="Privacy-Utility Trade-off",
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=10),
    )

    fig.add_hline(
        y=0.00,
        line_dash="dash",
        line_color="gray",
        annotation_text="aleatório (advantage = 0)",
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Modelo",
        hovermode="closest",
    )

    return fig