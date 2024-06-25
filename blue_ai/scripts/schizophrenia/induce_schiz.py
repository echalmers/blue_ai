import pandas as pd

from blue_ai.scripts.train_agents import load_trial, run_trial, save_trial
from blue_ai.scripts.constants import DATA_PATH
import matplotlib.pyplot as plt
from blue_ai.scripts.view_performance import PerformancePlotter


files = [
    # 'HealthyAgent_0.pkl',
    # 'SpineLossDepression_0.pkl',
    'SchizophrenicAgent_0.pkl'
]

fig, ax = plt.subplots(3, 2)

for i in range(len(files)):
    filename = files[i]
    results, agent, env = load_trial(DATA_PATH / filename)

    p1 = PerformancePlotter(results_dataframe=results)
    p1.plot_goals_per_episode(ax[i, 0], last_n_steps=10_000)
    plt.title(f'{filename}, before noise')
    plt.ylim((0, 2))

    noise_layer = agent.policy_net[1]
    assert hasattr(noise_layer, 'std')
    noise_layer.std = 0.25

    new_results, agent, env = run_trial(agent, env)
    p2 = PerformancePlotter(results_dataframe=new_results)
    p2.plot_goals_per_episode(ax[i, 1], last_n_steps=10_000)
    plt.ylim((0, 2))
    plt.title(f'{filename}, after noise')
    # plt.plot(new_results['cumulative_reward'])

    # combine results files
    new_filename = 'addNoise_' + filename
    save_trial(new_results, agent, env, DATA_PATH / new_filename)

# plt.legend(files)
plt.show()
