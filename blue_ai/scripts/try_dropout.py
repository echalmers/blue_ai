from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.dqn import DQN, SpinelossLayer
import matplotlib.pyplot as plt
from blue_ai.scripts.train_agents import run_trial
from torch import nn
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":

    # create the network and agent
    network = nn.Sequential(
        nn.Flatten(1, -1),
        SpinelossLayer(100, 10, dropout_rate=0),
        # nn.ReLU(),
        nn.Sigmoid(),
        nn.Linear(10, 3)
    )
    agent = DQN(
        network=network,
        input_shape=(4, 5, 5),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.01,
        sync_frequency=25,
        gamma=0.9,  # discount factor
        epsilon=0.05,  # random exploration rate
        batch_size=1500,
        weight_decay=0,  # we've been using 3e-3 for depression
    )

    # create the environment
    env = Image2VecWrapper(
        TransientGoals(
            render_mode="none", transient_reward=0.25, termination_reward=1
        )
    )

    # initial learning
    results1, agent, env = run_trial(
        trial_id='n/a',
        agent=agent,
        env=Image2VecWrapper(TransientGoals(render_mode="none")),
        steps=40_000,
        tbar=tqdm(total=40_000),
        save_activations=False,
    )

    # #depression on-set
    dropout = 0.2
    for layer in agent.policy_net.children():
        if isinstance(layer, SpinelossLayer):
            layer.set_dropout_rate(dropout)

    # depressed time
    results2, agent, env = run_trial(
        trial_id='n/a',
        agent=agent,
        env=env,
        steps=20_000,
        tbar=tqdm(total=20_000),
        save_activations=False,
    )
    results2['cumulative_reward'] += results1['cumulative_reward'].values[-1]
    results = pd.concat([results1, results2], ignore_index=True)

    # remove depression
    min_noise = .4
    max_noise = 1.0
    # dropout_removal = True
    for layer in agent.policy_net.children():
        if isinstance(layer, SpinelossLayer):
            # layer.set_noise_scale(min_noise, max_noise)
            # layer.add_noise()
            layer.set_dropout_rate(0)

    # learning after depression
    results3, agent, env = run_trial(
        trial_id='n/a',
        agent=agent,
        env=env,
        steps=50_000,
        tbar=tqdm(total=50_000),
        save_activations=False,
    )
    results3['cumulative_reward'] += results2['cumulative_reward'].values[-1]
    results = pd.concat([results, results3], ignore_index=True)

    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    plt.plot(results['cumulative_reward'])
    plt.title('cumulative reward')
    plt.sca(ax[1])
    plt.plot(results['reward'].rolling(500).mean())
    plt.title('reward-per-step (rolling avg)')
    plt.show()

