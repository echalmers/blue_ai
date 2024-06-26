from torch import nn
from blue_ai.agents.dqn import DQN
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


if __name__ == "__main__":

    rewards_table = pd.DataFrame(columns=["step", "cumulative reward"])
    goals_table = pd.DataFrame(columns=["goal type", "count"])

    for repetition in range(3):
        # a multi-layer network
        multilayer = nn.Sequential(
            nn.Flatten(1, -1), nn.Linear(100, 10), nn.Tanh(), nn.Linear(10, 3)
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
            weight_decay=0,  # we've been using 3e-3 for depression
        )

        # create the environment
        env = Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1
            )
        )  # set render mode to "human" to see the agent moving around
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
            print(step)
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

    sns.lineplot(rewards_table, x="step", y="cumulative reward", n_boot=5)
    plt.figure()
    sns.barplot(goals_table, x="goal type", y="count", n_boot=5)
    plt.show()
