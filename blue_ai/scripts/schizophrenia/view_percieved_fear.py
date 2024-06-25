from blue_ai.scripts.view_perceived_fear_2 import PerceivedFearPlotter
from blue_ai.scripts.constants import DATA_PATH
import matplotlib.pyplot as plt


plotter = PerceivedFearPlotter(
    save_filenames=[DATA_PATH / 'HealthyAgent_0.pkl',
                    DATA_PATH / 'addNoise_SchizophrenicAgent_0.pkl'
                    ]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_env_locations(ax[0])
plotter.plot_value_of_forward(ax[1])
plt.show()