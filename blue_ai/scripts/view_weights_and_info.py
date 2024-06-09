from blue_ai.scripts.weight_update import plot_weight_changes
from blue_ai.scripts.view_discounting import DiscountAndCorrelationPlotter
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from blue_ai.scripts.constants import FIGURE_PATH

if __name__ == "__main__":
    mosaic = """
    ab
    """
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(9, 4))

    plot_weight_changes(axes["a"])

    plotter = DiscountAndCorrelationPlotter()
    plotter.plot_neuron_correlations(axes["b"], palette=["skyblue", "salmon"])

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
    plt.savefig(FIGURE_PATH / "weight_changes.png", dpi=300)
    plt.show()
