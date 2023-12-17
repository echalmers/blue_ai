import pandas as pd

from blue_ai.scripts.train_agents import save_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression
from torch import nn
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

import os
import numpy as np


def plot_weight_changes():
    with open(os.path.join('.', 'data', 'weight_updates.pkl'), 'rb') as f:
        df = pd.DataFrame(pickle.load(f))
    print(df)

    df['cumulative reward'] = df.groupby(['rep', 'agent'])['reward'].transform(pd.Series.cumsum)
    df['smoothed loss'] = df.groupby(['rep', 'agent'])['loss'].transform(lambda x: x.rolling(8).mean())
    df['weight change per unit loss'] = df['weight_change'] / df['smoothed loss']

    ax = plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x='step', y='cumulative reward', hue='agent', n_boot=1)
    plt.xlabel('')

    plt.subplot(2, 1, 2, sharex=ax)
    sns.lineplot(data=df, x='step', y='weight_change', hue='agent', n_boot=1)
    plt.ylabel('weight change per unit loss')
    plt.legend([], [], frameon=False)

    plt.show()


if __name__ == '__main__':
    if os.path.exists(os.path.join('.', 'data', 'weight_updates.pkl')):
        plot_weight_changes()

    else:
        results = []
        for rep in range(5):
            print(rep)
            for agent in [HealthyAgent(), SpineLossDepression()]:

                env = Image2VecWrapper(TransientGoals(render_mode='none'))
                state, _ = env.reset()

                steps = 30_000
                steps_this_episode = 0
                reward_accumulator = 0

                for step in range(steps):
                    steps_this_episode += 1

                    # get & execute action
                    action = agent.select_action(np.expand_dims(state, 0))
                    new_state, reward, done, _, _ = env.step(action)
                    reward_accumulator += reward

                    # use this experience to update agent
                    old_weights = np.concatenate(
                        (agent.policy_net[1].weight.detach().flatten().numpy(), agent.policy_net[3].weight.detach().flatten().numpy())
                    )

                    loss = agent.update(state, action, reward, new_state, done=False)

                    if loss is not None:
                        new_weights = np.concatenate(
                            (agent.policy_net[1].weight.detach().flatten().numpy(), agent.policy_net[3].weight.detach().flatten().numpy())
                        )
                        changes = new_weights - old_weights
                        results.append({
                            'rep': rep,
                            'agent': agent.display_name,
                            'step': step,
                            'weight_change': np.mean(np.abs(changes)),
                            'loss': loss,
                            'reward': reward_accumulator
                        })
                        reward_accumulator = 0

                    # reset environment if done (ideally env would do this itself)
                    if done or steps_this_episode > 500:
                        state, _ = env.reset()
                        steps_this_episode = 0
                    else:
                        state = new_state

        with open(os.path.join('.', 'data', 'weight_updates.pkl'), 'wb') as f:
            pickle.dump(
                pd.DataFrame(results).to_dict(orient='list'),
                f
            )
        plot_weight_changes()

    exit()





# for dropout in [0, 50]:
#
#
#     env = Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=0.25, termination_reward=1))
#
#     # a multi-layer network
#     outputs = 3
#     multilayer = nn.Sequential(
#         nn.Flatten(1, -1),
#         nn.Dropout(p=(dropout / 100)),
#         # LostSpinesLayer(in_features=147, p=dropout / 100),
#         nn.Linear(147, 10),
#         # ConnectionDropout(147, 10, p=dropout / 100),
#         nn.Tanh(),
#         nn.Dropout(p=(dropout / 100)),
#         # StaticDropout(in_features=10, p=dropout / 100),
#         nn.Linear(10, outputs)
#     )
#     agent = DQN(
#             network=multilayer,  # supply either network here
#             input_shape=(3, 7, 7),
#             replay_buffer_size=10000,
#             update_frequency=5,
#             lr=0.005,
#             sync_frequency=25,
#             gamma=0.9, epsilon=0.05,
#             batch_size=1500
#         )
#
#     # setup the environment
#     state, _ = env.reset()
#
#     weight_changes = []
#     losses = []
#     grads = []
#     steps_this_episode = 0
#     for step in range(15000):
#
#         # get & execute action
#         action = agent.select_action(np.expand_dims(state, 0))
#         new_state, reward, done, _, _ = env.step(action)
#
#         # use this experience to update agent
#         old_weights = np.concatenate((agent.policy_net[2].weight.detach().flatten().numpy(), agent.policy_net[5].weight.detach().flatten().numpy()))
#         # if reward > 0:
#             # loss = agent.update_single(state, action, reward, new_state, done=False)
#             # losses.append(loss)
#
#         agent.update(state, action, reward, new_state, done=False)
#
#         new_weights = np.concatenate((agent.policy_net[2].weight.detach().flatten().numpy(), agent.policy_net[5].weight.detach().flatten().numpy()))
#         changes = new_weights - old_weights
#         if not all(new_weights == old_weights):
#             weight_changes.append(np.mean(np.abs(changes)))
#
#         if agent.policy_net[2].weight.grad is not None:
#             grads.append(
#                 np.mean(np.abs(
#                     np.concatenate((agent.policy_net[2].weight.grad.detach().flatten().numpy(),
#                                     agent.policy_net[5].weight.grad.detach().flatten().numpy()))
#                 ))
#             )
#
#         # reset environment if done (ideally env would do this itself)
#         if done or steps_this_episode > 500:
#             state, _ = env.reset()
#             steps_this_episode = 0
#         else:
#             state = new_state
#
#     ax = plt.subplot(2,1,1)
#     plt.plot(weight_changes)
#     plt.subplot(2,1,2, sharex=ax)
#     plt.plot(losses)
#
# plt.legend(['healthy', 'depressed'])
# # plt.ylabel('mean absolute weight change')
# plt.xlabel('step #')
# plt.show()