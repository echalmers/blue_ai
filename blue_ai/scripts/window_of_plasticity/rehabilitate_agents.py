from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH
from copy import deepcopy
from blue_ai.scripts.schizophrenia.train_interpretation_models import RepresentationProbe
from torch import nn
import time


import torch
import pandas as pd
import numpy as np


good_env = Image2VecWrapper(TransientGoals(render_mode="none", n_transient_obstacles=1, n_transient_goals=3))
good_env.max_reward = good_env.n_transient_goals * 0.25 + 1
bad_env = Image2VecWrapper(TransientGoals(render_mode="none", n_transient_obstacles=3, n_transient_goals=1))
bad_env.max_reward = bad_env.n_transient_goals * 0.25 + 1


class RepresentationError:
    def __init__(self, agent):
        self.probe = RepresentationProbe(agent)

    def __call__(self, agent, state):
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32), device=agent.device)
        recon = self.probe.get_reconstructions(observations=state)[1][0]
        mse = nn.MSELoss()(recon, state)
        return mse.item()


all_results = pd.DataFrame()
start = time.time()
for rep in range(2):
    for agent_object in [HealthyAgent(), SpineLossDepression(3e-3)]:
        for env1, env1_name in [(good_env, 'good'), (bad_env, 'bad')]:
            print(f'rep {rep} ({round(time.time() - start, 0)}s): {agent_object} {env1_name}')

            # initial training
            results1, agent, _ = run_trial(
                trial_id=rep,
                agent=deepcopy(agent_object),
                env=env1,
                steps=35_000,
            )

            # fit representation model
            print('     fitting representation model')
            mse_measurer = RepresentationError(agent)

            # a bit more training, this time with representation error measurements
            results2, agent, _ = run_trial(
                trial_id=rep,
                agent=agent,
                env=env1,
                steps=5_000,
                callbacks=[{'name': 'reconstruction_mse', 'function': mse_measurer, 'period': 20}],
                learning=True
            )

            # remove weight decay
            agent.optimizer = torch.optim.Adam(
                agent.policy_net.parameters(), lr=agent.lr, weight_decay=0
            )

            for env2, env2_name in [(good_env, 'good'), (bad_env, 'bad')]:
                results3, _, _ = run_trial(
                    trial_id=rep,
                    agent=agent,
                    env=env2,
                    steps=25_000,
                    callbacks=[{'name': 'reconstruction_mse', 'function': mse_measurer, 'period': 10}]
                )

                results1['env_max_reward'] = env1.max_reward
                results2['env_max_reward'] = env1.max_reward
                results3['env_max_reward'] = env2.max_reward
                results1['stage'] = 0
                results2['stage'] = 0
                results3['stage'] = 1

                new_results = pd.concat([results1, results2, results3], ignore_index=True)
                new_results["cumulative_reward"] = new_results["reward"].cumsum()
                new_results["step"] = np.arange(new_results.shape[0])
                new_results['path'] = env1_name + env2_name

                all_results = pd.concat([all_results, new_results], ignore_index=True)

# compute avg reward for healthy agent, for reference
healthy_data = all_results[all_results['agent'] == 'HealthyAgent']
healthy_mean = healthy_data.groupby(['path', 'stage'])['reward'].mean().reset_index()
all_results = pd.merge(
    left=all_results,
    right=healthy_mean,
    on=['path', 'stage'],
    suffixes=('', '_healthy')
)

all_results.to_csv(DATA_PATH / 'rehabilitation_tests.csv', index=False)
