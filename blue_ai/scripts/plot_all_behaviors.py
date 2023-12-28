import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from view_performance import PerformancePlotter
from view_discounting import DiscountAndCorrelationPlotter
from view_perceived_fear_2 import PerceivedFearPlotter
from view_perceived_value import PerceivedValuePlotter
from blue_ai.agents.agent_classes import SpineLossDepression

SpineLossDepression.display_name = 'simulated\nspine loss'

performance_plotter = PerformancePlotter()
discount_plotter = DiscountAndCorrelationPlotter()

mosaic = '''
a.b
c.d
'''

fig, axes = plt.subplot_mosaic(mosaic, figsize=(11, 8), width_ratios=[10, 0, 10])  #, height_ratios=[47.5, 5, 47.5])
performance_plotter.plot_sample_env(axes['a'])
performance_plotter.plot_goals_per_episode(axes['b'])
discount_plotter.plot_inferred_discount(axes['d'], palette=['skyblue', 'salmon'])
# discount_plotter.plot_neuron_correlations(axes['e'], palette=['skyblue', 'salmon'])

performance_plotter.plot_learning_curves(axes['c'], palette=['skyblue', 'salmon'])
plt.xticks([0, 20000])
plt.text(x=12_000, y=375, s='healthy', c='blue')
plt.text(x=16_000, y=125, s='simulated\nspine loss', c='red')
plt.legend([], [], frameon=False)

for label, ax in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', weight='bold', va='bottom', fontfamily='serif')

plt.tight_layout()
plt.savefig('img/performance_results.png', dpi=300)


fear_plotter = PerceivedFearPlotter()
value_plotter = PerceivedValuePlotter()

mosaic = '''
e.f
...
g.h
'''
fig2, axes2 = plt.subplot_mosaic(mosaic, figsize=(9, 9), width_ratios=[47.5, 5, 47.5], height_ratios=[47.5, 5, 47.5])
value_plotter.plot_perceived_value_env(axes2['e'])
value_plotter.plot_perceived_value(axes2['f'])
fear_plotter.plot_env_locations(axes2['g'])
fear_plotter.plot_value_of_forward(axes2['h'])
plt.plot([0, 1], [0.5, 0.5], color='grey', lw=1, transform=plt.gcf().transFigure, clip_on=False)

plt.tight_layout()
plt.savefig('img/value_fear.png', dpi=300)
plt.show()