import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.transforms as mtransforms

from view_perceived_value import PerceivedValuePlotter
from view_discounting import DiscountAndCorrelationPlotter
from view_performance import PerformancePlotter
from view_perceived_fear_2 import PerceivedFearPlotter
from compare_exploration import plot_kl_divergence
from blue_ai.agents.agent_classes import (
    HealthyAgent,
    SpineLossDepression,
    ScaledTargets,
    ContextDependentLearningRate,
    HighDiscountRate,
    HighExploration,
)


SpineLossDepression.display_name = "simulated\nspine loss"
ScaledTargets.display_name = "reduced RPE"
ContextDependentLearningRate.display_name = "higher negative\nlearning rate"
HighDiscountRate.display_name = "high\ndiscounting"
HighExploration.display_name = "high\nexploration"

all_agent_classes = [
    HealthyAgent,
    SpineLossDepression,
    ScaledTargets,
    ContextDependentLearningRate,
    HighDiscountRate,
    HighExploration,
]
extra_colors = sns.color_palette()[1:]
del extra_colors[2]
all_colors = ["skyblue", "salmon"] + extra_colors


mosaic = """
abbb
ccdd
efff
gg..
"""
fig, axes = plt.subplot_mosaic(mosaic, figsize=(15, 12))

plot_kl_divergence(axes["g"], palette=all_colors)
plt.title("KL divergence from healthy probability distribution over actions")
plt.xlabel("")

plotter = PerceivedValuePlotter(agent_classes=all_agent_classes)
plotter.plot_perceived_value_env(axes["a"])
plotter.plot_perceived_value(axes["b"], palette=all_colors)

plotter = PerformancePlotter(agent_classes=all_agent_classes)
plotter.plot_goals_per_episode(axes["c"])
plt.ylim((0, 2.75))
plt.legend(ncol=3, loc="upper right")

plotter = DiscountAndCorrelationPlotter(agent_classes=all_agent_classes)
plotter.plot_inferred_discount(axes["d"], palette=all_colors)
plt.title("inferred discount factor (lower means more discounting)")

for agent in all_agent_classes:
    agent.display_name = agent.display_name.replace("\n", " ")
plotter = PerceivedFearPlotter(agent_classes=all_agent_classes)
plotter.plot_env_locations(axes["e"])
plotter.plot_value_of_forward(axes["f"], palette=all_colors)
plt.ylim((-0.6, 1.6))
plt.plot([1, 4], [1, 1], c="grey", linestyle=":")
plt.title("perceived value of moving forward (normalized to position 1 value)")
plt.legend(ncol=3)


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
plt.savefig("img/alternatives.png", dpi=300)
plt.show()
