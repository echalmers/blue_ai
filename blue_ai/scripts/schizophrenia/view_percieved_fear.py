from blue_ai.scripts.view_perceived_fear_2 import PerceivedFearPlotter
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS, _CURRENT_DIR
from blue_ai.envs.transient_goals import Actions
import matplotlib.pyplot as plt


plotter = PerceivedFearPlotter(
    save_filenames=[
        filename
        for trial in range(N_TRIALS)
        for filename in [
            DATA_PATH / f'HealthyAgent_{trial}.pkl',
            DATA_PATH / f'addNoise_SchizophrenicAgent_{trial}.pkl'
        ]
    ],
    actions_included=[Actions.right, Actions.forward]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_env_locations(ax[0])
plotter.plot_value_of_forward(ax[1], palette=['blue', 'orange'])
ax[1].set_title('perceived value of actions\n(normalized to position 1 value)')

plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'fear.png', dpi=400)
plt.show()