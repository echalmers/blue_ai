from blue_ai.scripts.view_discounting import DiscountAndCorrelationPlotter
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS, _CURRENT_DIR
import matplotlib.pyplot as plt



plotter = DiscountAndCorrelationPlotter(
    save_filenames=[
        filename
        for trial in range(N_TRIALS)
        for filename in [
            # DATA_PATH / 'addNoise_HealthyAgent_0.pkl',
            DATA_PATH / f'HealthyAgent_{trial}.pkl',
            DATA_PATH / f'addNoise_SchizophrenicAgent_{trial}.pkl',
            # DATA_PATH / f'SpineLossDepression_{trial}.pkl'
        ]
    ]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_env_locations(ax[0])
plotter.plot_inferred_discount(ax[1])
# plotter.plot_neuron_correlations(ax[2])

plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'discounting.png', dpi=400)
plt.show()
