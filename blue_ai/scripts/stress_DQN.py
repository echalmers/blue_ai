import torch
from torch import nn
from torch import optim
import random
from matplotlib import pyplot as plt
import numpy as np
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.dqn import DQN
import pandas as pd
from constants import DATA_PATH

results = pd.DataFrame()

for rep in range(10):
    for stress_length in [10, 50]:

        # Set general parameters
        gamma = 0.9
        epsilon = 0.01
        learning_rate = 0.01
        weight_decay = 0
        episodes_per_trial = [500, stress_length, 550 - stress_length]

        weight_decay_multiplier = 0.005

        # Initialize reward variables
        total_reward = 0
        expected_reward_longterm = 0
        expected_reward_shortterm = 0
        diff = 0
        reward_responsiveness_shortterm = 0.001
        reward_responsiveness_longterm = 0.0001

        # Initializing reward lists and weight decay list
        expected_reward_longterm_list = []
        expected_reward_shortterm_list = []
        expected_reward_difference_list = []
        total_reward_list = []
        weight_decay_list = []
        average_reward = []


        # Initialize network
        multilayer = nn.Sequential(
                    nn.Flatten(1, -1), nn.Linear(100, 10), nn.Tanh(), nn.Linear(10, 3)
                )

        # instantiate the agent
        agent = DQN(
            network=multilayer,
            input_shape=(4, 5, 5),
            replay_buffer_size=10000,
            update_frequency=5,
            lr=learning_rate,
            sync_frequency=25,
            gamma=gamma,  # discount factor
            epsilon=epsilon,  # random exploration rate
            batch_size=1500,
            weight_decay=weight_decay,  # we've been using 3e-3 for depression
        )

        for trial in range(3):

            if trial != 1:
                # create the environment
                env = Image2VecWrapper(
                    TransientGoals(
                        render_mode="none", transient_reward=0.25, termination_reward=1)  #, transient_locations=[(3, 3), (5, 4)])
                )  # set render mode to "human" to see the agent moving around
            else:
                # create the environment
                env = Image2VecWrapper(
                    TransientGoals(
                        render_mode="none",
                        transient_reward=0.25,
                        termination_reward=1,
                        n_transient_obstacles=10,
                        n_transient_goals=0
                        # transient_locations=[(3, 3), (5, 4)]
                    )
                )  # set render mode to "human" to see the agent moving around



            steps_list = []

            # Training loop
            for episode in range(episodes_per_trial[trial]):
                # Reset the environment and get the initial state
                state = env.reset()[0]

                # Reset steps
                steps = 0

                # Add a maximum of steps within the episodes to be faster
                terminated = truncated = False

                #if episode % 10 == 0 and trial == 1:
                #    reward_responsiveness_shortterm = reward_responsiveness_shortterm - 0.00001
                #    reward_responsiveness_longterm = reward_responsiveness_longterm - 0.000001

                # if len(weight_decay_list) > 1: #and trial != 0:
                #     #if (weight_decay_list[-1] > weight_decay_list[-2] and episode % 15 == 0):
                #     if (trial == 1 and episode % 15 == 0):
                #         reward_responsiveness_shortterm = reward_responsiveness_shortterm - 0.00001
                #         reward_responsiveness_longterm = reward_responsiveness_longterm - 0.000001
                #         learning_rate = learning_rate - 0.002
                #
                #     elif (trial == 2 and episode % 15 == 0):
                #     #elif (weight_decay_list[-1] < weight_decay_list[-2] and episode % 15 == 0):
                #         reward_responsiveness_shortterm = reward_responsiveness_shortterm + 0.00001
                #         #reward_responsiveness_shortterm = min(reward_responsiveness_shortterm, 0.001)
                #         reward_responsiveness_longterm = reward_responsiveness_longterm + 0.000001
                #         #reward_responsiveness_longterm = min(reward_responsiveness_longterm, 0.0001)
                #         learning_rate = learning_rate + 0.001
                #
                #     agent.lr = learning_rate

                while not (terminated or truncated) and steps < 200:
                    # Choose action using epsilon-greedy strategy
                    action = agent.select_action(state)

                    # Take the action and observe the next state and reward
                    next_state, reward, terminated, truncated, _ = env.step(action)

                    # Accumulate the total reward and memory
                    total_reward += reward

                    expected_reward_longterm = expected_reward_longterm * (1 - reward_responsiveness_longterm) + reward * reward_responsiveness_longterm
                    #expected_reward_longterm = expected_reward_longterm * (1 - 0.0001) + reward * 0.0001
                    #expected_reward_longterm = min(0.08, expected_reward_longterm)
                    #expected_reward_longterm = max(-0.08, expected_reward_longterm)
                    expected_reward_shortterm = expected_reward_shortterm * (1 - reward_responsiveness_shortterm) + reward * reward_responsiveness_shortterm
                    #expected_reward_shortterm = expected_reward_shortterm * (1 - 0.001) + reward * 0.001
                    #expected_reward_shortterm = min(0.08, expected_reward_shortterm)
                    #expected_reward_shortterm = max(-0.08, expected_reward_shortterm)

                    # Update the Q-network with the observed transition
                    agent.update(
                        state=state,
                        new_state=next_state,
                        reward=reward,
                        done=terminated,
                        action=action,
                    )

                    steps += 1

                    # Move to the next state
                    state = next_state

                expected_reward_longterm_list.append(expected_reward_longterm)
                expected_reward_shortterm_list.append(expected_reward_shortterm)
                diff = 0.5 * diff + 0.5 * (expected_reward_longterm - expected_reward_shortterm)
                diff = min(0.06, diff)
                diff = max(-0.06, diff)
                expected_reward_difference_list.append(diff)

                # Track cumulative and average rewards
                total_reward_list.append(total_reward)
                average_reward.append(total_reward / (episode + 1))

                # Calculate weight decay
                #if expected_reward_difference_list[-1] > 0:
                #    weight_decay += expected_reward_difference_list[-1] * 0.001
                #else:
                #    weight_decay += expected_reward_difference_list[-1] * 0.0005
                #if trial != 0:

                # if expected_reward_difference_list[-1] > 0 and episode % 20 == 0:
                # #if trial == 1 and episode % 10 == 0:
                #     weight_decay_multiplier += 0.00001
                #
                # #elif episode % 25 == 0:
                # elif trial == 2 and episode % 10 == 0:
                #     weight_decay_multiplier -= 0.0005
                #
                # weight_decay_multiplier = max(0.0005, weight_decay_multiplier)

                weight_decay += expected_reward_difference_list[-1] * weight_decay_multiplier

                # if expected_reward_shortterm - expected_reward_longterm < 0:
                #    weight_decay += expected_reward_difference_list[-1] * 0.0005
                # else:
                #    weight_decay += expected_reward_difference_list[-1] * 0.0001

                weight_decay = max(0, weight_decay)
                #weight_decay = min(0.01, weight_decay)
                weight_decay_list.append(weight_decay)
                if weight_decay != agent.optimizer.param_groups[0]['weight_decay']:
                    agent.optimizer.param_groups[0]['weight_decay'] = weight_decay # seems to be faster than re-initializing the opitmizer

                steps_list.append(steps)

                if episode % 100 == 0:
                    print(f"episode {episode}")

                if trial == 1 and episode == 499:
                    print(f"weight decay multiplier {weight_decay_multiplier}")

                if trial == 2 and episode == 499:
                    print(f"weight decay multiplier {weight_decay_multiplier}")

            # Calculate average reward
            average_expected_reward = sum(expected_reward_difference_list) / len(expected_reward_difference_list)

            # Print the total and average rewards
            print(f"Total reward: {total_reward}")
            print(f"Average reward: {average_reward[-1]}")
            print(f"Average Expected Reward {average_expected_reward}")
            print(f"Average steps {sum(steps_list) / len(steps_list)} ")

        # Smooth reward lists using rolling mean
        #expected_reward_longterm_list = pd.Series(expected_reward_longterm_list).rolling(window=50).mean()
        #expected_reward_shortterm_list = pd.Series(expected_reward_shortterm_list).rolling(window=50).mean()
        #weight_decay_list = pd.Series(weight_decay_list).rolling(window=50).mean()

        these_results = pd.DataFrame()
        these_results['rep'] = rep
        these_results['episode'] = np.arange(len(total_reward_list))
        these_results['stress_length'] = stress_length
        these_results['cumulative_reward'] = total_reward_list
        these_results['expected_shortterm'] = expected_reward_shortterm_list
        these_results['expected_longterm'] = expected_reward_longterm_list
        these_results['diff'] = expected_reward_difference_list
        these_results['weight_decay'] = weight_decay_list
        results = pd.concat((results, these_results))

results.to_csv(DATA_PATH / 'stress.csv', index=False)