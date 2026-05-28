from blue_ai.scripts.view_perceived_value import PerceivedValuePlotter
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS, _CURRENT_DIR
import matplotlib.pyplot as plt


plotter = PerceivedValuePlotter(
    save_filenames=[
        filename
        for i in range(N_TRIALS)
        for filename in [
            DATA_PATH / f'HealthyAgent_{i}.pkl',
            DATA_PATH / f'addNoise_SchizophrenicAgent_{i}.pkl'
            ]
    ]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_perceived_value_env(ax[0])
plotter.plot_perceived_value(ax[1], palette=['skyblue', 'orange'])

plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'anhedonia.png', dpi=400)
plt.show()