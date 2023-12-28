from weight_update import plot_weight_changes
from view_discounting import DiscountAndCorrelationPlotter
import matplotlib.pyplot as plt


plt.figure(figsize=(9,4))
ax = plt.subplot(1,2,1)
plot_weight_changes(ax)

ax = plt.subplot(1, 2, 2)
plotter = DiscountAndCorrelationPlotter()
plotter.plot_neuron_correlations(ax, palette=['skyblue', 'salmon'])

plt.tight_layout()
plt.savefig('img/weight_changes.png', dpi=300)
plt.show()