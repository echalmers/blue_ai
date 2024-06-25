from torch import nn
from blue_ai.agents.dqn import DQN
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def run_trial(agent, env):

    rewards_table = pd.DataFrame(columns=["step", "cumulative reward"])
    goals_table = pd.DataFrame(columns=["goal type", "count"])

    for repetition in trange(3):
        state, _ = env.reset()

        # set up an array and other variables to store results
        STEPS = 30_000
        steps_this_episode = 0
        rewards = np.zeros(STEPS)

        num_required_goals = 0
        num_optional_goals = 0
        num_lava = 0

        # training loop
        for step in range(STEPS):
            steps_this_episode += 1

            # get & execute action
            action = agent.select_action(state)
            new_state, reward, done, _, _ = env.step(action)
            rewards[step] = reward

            # record goals found
            num_required_goals += reward == 1
            num_optional_goals += 0 < reward < 1
            num_lava += reward < 0

            # update the agent
            agent.update(
                state=state,
                new_state=new_state,
                reward=reward,
                done=done,
                action=action,
            )

            # reset the environment if goal reached
            if done or steps_this_episode > 500:
                state, _ = env.reset()
                steps_this_episode = 0
            else:
                state = new_state

        rewards_table = pd.concat(
            (
                rewards_table,
                pd.DataFrame(
                    {"step": np.arange(STEPS), "cumulative reward": rewards.cumsum()}
                ),
            ),
            ignore_index=True,
        )
        goals_table = pd.concat(
            (
                goals_table,
                pd.DataFrame(
                    {
                        "goal type": ["required", "optional", "lava"],
                        "count": [num_required_goals, num_optional_goals, num_lava],
                    }
                ),
            )
        )
    return rewards_table, goals_table


if __name__ == "__main__":

    # a multi-layer network
    multilayer = nn.Sequential(
        nn.Flatten(1, -1), nn.Linear(100, 10), nn.Sigmoid(), nn.Linear(10, 3)
    )

    # instantiate the agent
    agent = DQN(
        network=multilayer,
        input_shape=(4, 5, 5),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.01,
        sync_frequency=25,
        gamma=0.9,  # discount factor
        epsilon=0.05,  # random exploration rate
        batch_size=1500,
        weight_decay=0,
    )

    env = Image2VecWrapper(
        TransientGoals(render_mode="none", transient_penalty=-0.1, n_transient_goals=5)
    )

    rewards_table, goals_table = run_trial(agent, env)

    rewards_table["agent"] = "Healthy"
    goals_table["agent"] = "Health"

    # a multi-layer network
    multilayer = nn.Sequential(
        nn.Flatten(1, -1), nn.Linear(100, 10), nn.Sigmoid(), nn.Linear(10, 3)
    )

    # instantiate the agent
    agent = DQN(
        network=multilayer,
        input_shape=(4, 5, 5),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.01,
        sync_frequency=25,
        gamma=0.9,  # discount factor
        epsilon=0.05,  # random exploration rate
        batch_size=1500,
        weight_decay=1e-3,
    )

    env = Image2VecWrapper(
        TransientGoals(render_mode="none", transient_penalty=-0.1, n_transient_goals=5)
    )

    r, g = run_trial(agent, env)

    r["agent"] = "SpineLoss"
    g["agent"] = "SpineLoss"

    rewards_table = pd.concat([rewards_table, r])
    goals_table = pd.concat([goals_table, g])

    fig, axes = plt.subplots(1, 2)
    sns.lineplot(
        rewards_table,
        x="step",
        y="cumulative reward",
        hue="agent",
        n_boot=5,
        ax=axes[0],
    )
    sns.barplot(
        goals_table,
        x="goal type",
        y="count",
        hue="agent",
        n_boot=5,
        ax=axes[1],
    )
    plt.savefig("sample_good_og.png")
