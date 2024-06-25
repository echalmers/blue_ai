from blue_ai.scripts.view_perceived_value import PerceivedValuePlotter
from blue_ai.scripts.constants import DATA_PATH
import matplotlib.pyplot as plt


plotter = PerceivedValuePlotter(
    save_filenames=[DATA_PATH / 'HealthyAgent_0.pkl',
                    DATA_PATH / 'addNoise_SchizophrenicAgent_0.pkl'
                    ]
)

fig, ax = plt.subplots(1, 2)
plotter.plot_perceived_value_env(ax[0])
plotter.plot_perceived_value(ax[1])
plt.show()