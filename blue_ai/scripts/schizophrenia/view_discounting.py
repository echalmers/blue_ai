from blue_ai.scripts.view_discounting import DiscountAndCorrelationPlotter
from blue_ai.scripts.constants import DATA_PATH
import matplotlib.pyplot as plt

plotter = DiscountAndCorrelationPlotter(
    save_filenames=[
        # DATA_PATH / 'addNoise_HealthyAgent_0.pkl',
        DATA_PATH / 'HealthyAgent_0.pkl',
        DATA_PATH / 'addNoise_SchizophrenicAgent_0.pkl',
        DATA_PATH / 'SpineLossDepression_0.pkl',
    ]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_env_locations(ax[0])
plotter.plot_inferred_discount(ax[1])
# plotter.plot_neuron_correlations(ax[2])
plt.show()
