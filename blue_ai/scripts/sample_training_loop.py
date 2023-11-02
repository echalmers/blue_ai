from torch import nn
from blue_ai.agents.dqn import DQN
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


rewards_table = pd.DataFrame(columns=['step', 'cumulative reward'])
goals_table = pd.DataFrame(columns=['goal type', 'count'])

for _ in range(10):
    # a multi-layer network
    multilayer = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(100, 10),
        nn.Tanh(),
        nn.Linear(10, 3)
    )

    # instantiate the agent
    agent = DQN(
            network=multilayer,
            input_shape=(4, 5, 5),
            replay_buffer_size=10000,
            update_frequency=5,
            lr=0.005,
            sync_frequency=25,
            gamma=0.2,  # discount factor
            epsilon=0.05,  # random exploration rate
            batch_size=1500,
            weight_decay=0,  # we've been using 2.5e-3 for depression
    )

    # 3 possible other ways to simulate depression:
    # - scale down reward-prediction-error (Dopamine) by multiplying line 146 of dqn.py by something less than 1
    # - shift down reward_prediction_error (also on line 146) by subtracting instead of multiplying
    # - use a smaller discount factor (this *should* be the same or similar to the downscaling, could try to match the effects)


    # create the environment
    env = Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=0.25, termination_reward=1))  # set render mode to "human" to see the agent moving around
    state, _ = env.reset()

    # set up an array and other variables to store results
    STEPS = 30_000
    rewards = np.zeros(STEPS)
    num_required_goals = 0
    num_optional_goals = 0
    num_lava = 0


    # training loop
    for step in range(STEPS):
        print(step)

        # get & execute action
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)
        rewards[step] = reward

        # record goals found
        num_required_goals += reward == 1
        num_optional_goals += 0 < reward < 1
        num_lava += reward < 0

        # update the agent
        agent.update(state=state, new_state=new_state, reward=reward, done=done, action=action)

    rewards_table = pd.concat((rewards_table, pd.DataFrame({'step': np.arange(STEPS), 'cumulative reward': rewards.cumsum()})), ignore_index=True)
    goals_table = pd.concat((goals_table, pd.DataFrame({'goal type': ['required', 'optional', 'lava'], 'count': [num_required_goals, num_optional_goals, num_lava]})))

sns.lineplot(rewards_table, x='step', y='cumulative reward', n_boot=5)
plt.figure()
sns.barplot(goals_table, x='goal type', y='count', n_boot=5)
plt.show()