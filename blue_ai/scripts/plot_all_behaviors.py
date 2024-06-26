import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from blue_ai.scripts.view_performance import PerformancePlotter
from blue_ai.scripts.view_discounting import DiscountAndCorrelationPlotter
from blue_ai.scripts.view_perceived_fear_2 import PerceivedFearPlotter
from blue_ai.scripts.view_perceived_value import PerceivedValuePlotter
from blue_ai.agents.agent_classes import SpineLossDepression
from blue_ai.scripts.constants import FIGURE_PATH


if __name__ == "__main__":

    SpineLossDepression.display_name = "simulated\nspine loss"

    performance_plotter = PerformancePlotter()
    discount_plotter = DiscountAndCorrelationPlotter()
    fear_plotter = PerceivedFearPlotter()
    value_plotter = PerceivedValuePlotter()

    mosaic = """
    abbc
    ...c
    de.c
    """

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(11, 8),
        width_ratios=[10, 10, 1, 10],
        height_ratios=[47.5, 5, 47.5],
    )
    performance_plotter.plot_sample_env(axes["a"])
    performance_plotter.plot_goals_per_episode(axes["b"])
    value_plotter.plot_perceived_value_env(axes["d"])
    value_plotter.plot_perceived_value(axes["e"])
    performance_plotter.plot_learning_curves(axes["c"], palette=["skyblue", "salmon"])
    plt.xticks([0, 20000])
    plt.text(x=10_000, y=375, s="healthy", c="blue")
    plt.text(x=14_000, y=200, s="simulated\nspine loss", c="red")
    plt.legend([], [], frameon=False)

    for label, ax in axes.items():
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize="large",
            weight="bold",
            va="bottom",
            fontfamily="serif",
        )

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "performance_results.png", dpi=300)

    mosaic = """
    a.b
    ...
    c.d
    """
    fig2, axes2 = plt.subplot_mosaic(
        mosaic,
        figsize=(9, 9),
        width_ratios=[47.5, 5, 47.5],
        height_ratios=[47.5, 5, 47.5],
    )
    fear_plotter.plot_env_locations(axes2["c"])
    fear_plotter.plot_value_of_forward(axes2["d"], palette=["skyblue", "salmon"])
    discount_plotter.plot_env_locations(axes2["a"])
    discount_plotter.plot_inferred_discount(axes2["b"], palette=["skyblue", "salmon"])
    plt.plot(
        [0, 1],
        [0.5, 0.5],
        color="grey",
        lw=1,
        transform=plt.gcf().transFigure,
        clip_on=False,
    )

    for label, ax in axes2.items():
        # label physical distance to the left and up:
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize="large",
            weight="bold",
            va="bottom",
            fontfamily="serif",
        )

    plt.tight_layout()
    plt.savefig(FIGURE_PATH / "value_fear.png", dpi=300)
    plt.show()
