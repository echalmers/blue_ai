from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


for rep in range(10):
    results1, agent, env = run_trial(
        agent=HealthyAgent(),
        env=Image2VecWrapper(TransientGoals(render_mode='none')),
        steps=20_000,
    )

    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=agent.lr, weight_decay=5e-3)

    results2, agent, env = run_trial(
        agent=agent,
        env=env,
        steps=20_000,
    )

    agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=agent.lr, weight_decay=0)

    results3, agent, env = run_trial(
        agent=agent,
        env=env,
        steps=20_000,
    )

    results = pd.concat([results1, results2, results3], ignore_index=True)
    results['cumulative_reward'] = results['reward'].cumsum()
    results['step'] = np.arange(results.shape[0])
    save_trial(results, agent, env, filename=os.path.join('.', 'data', f'rehabilitate_{rep}.pkl'))

plt.plot(results['cumulative_reward'])
plt.show()
