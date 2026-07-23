import plotly.express as px

from visualization.common import build_tradeoff_points


def plot_privacy_utility_tradeoff(utility_results, attack_results):
    """
    Trade-off direto Utilidade X Vazamento.

    X = perda relativa de utilidade
    Y = advantage
    """

    tradeoff_df = build_tradeoff_points(
        utility_results=utility_results,
        attack_results=attack_results,
    )

    fig = px.line(
        tradeoff_df,
        x="utility_loss",
        y="advantage",
        color="model",
        text="epsilon",
        hover_data={
            "epsilon": True,
            "test_mae": ":.0f",
            "attack_acc": ":.3f",
            "utility_loss": ":.3f",
            "advantage": ":.3f",
        },
        labels={
            "utility_loss": "Perda relativa de utilidade",
            "advantage": "Advantage",
            "model": "Modelo",
        },
        title="Privacy–Utility Trade-off",
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=12),
    )

    fig.add_hline(
        y=0.00,
        line_dash="dash",
        line_color="gray",
    )

    fig.add_hline(
        y=0.03,
        line_dash="dot",
        line_color="orange",
    )

    fig.add_hline(
        y=0.05,
        line_dash="dot",
        line_color="red",
    )

    fig.update_yaxes(
        range=[-0.02, 0.057],
        title="Advantage",
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Modelo",
        hovermode="closest",
    )

    return fig