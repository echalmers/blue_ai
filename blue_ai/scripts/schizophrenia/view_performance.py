from blue_ai.scripts.train_agents import load_trial, run_trial, save_trial
from blue_ai.scripts.constants import DATA_PATH
import matplotlib.pyplot as plt
from blue_ai.scripts.view_performance import PerformancePlotter


files = [
    'HealthyAgent_0.pkl',
    'SpineLossDepression_0.pkl',
    'SchizophrenicAgent_0.pkl'
]

fig, ax = plt.subplots(3, 2)

for i in range(len(files)):
    filename = files[i]
    results, _, _ = load_trial(DATA_PATH / filename)

    p1 = PerformancePlotter(results_dataframe=results)
    p1.plot_goals_per_episode(ax[i, 0], last_n_steps=10_000)
    plt.title(f'{filename}, before noise')
    plt.ylim((0, 2))

    new_results, _, _ = load_trial(DATA_PATH / ('addNoise_' + filename))
    p2 = PerformancePlotter(results_dataframe=new_results)
    p2.plot_goals_per_episode(ax[i, 1], last_n_steps=10_000)
    plt.ylim((0, 2))
    plt.title(f'{filename}, after noise')

fig = plt.figure()


plt.show()
