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

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create neural network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Define the layers of the network
        self.fc0 = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(state_dim, 10)      # Input layer (100 neurons)
        self.fc2 = nn.Linear(10, action_dim)      # Output layer (3 neurons)

    def forward(self, x):
        # Apply ReLu activation function after the input and hidden layer
        x = self.fc0(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x # tensor of size 3, each element is an action_value


# Epsilon-greedy policy for action selection
def get_action(network, state, epsilon):

    # Convert state to one_hot encoded tensor
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # With probability epsilon, choose random action
    if random.random() < epsilon:
        action = np.random.randint(0,3)  # Random action
    else:
        # Otherwise, choose the action with the highest value
        with torch.no_grad():
            action_values = network(state)
            # action = action_values.max(1)[1].item()
            action = torch.argmax(action_values).item()
    return action


# Function to update the Q-network
def update_network(network, state, action, reward, next_state, gamma, weight_decay):
    # Convert the current and next state to one-hot encoded tensors
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

    # Get the current action values from the Q-network
    current_values = network(state)  # Shape: [1, num_actions]
    # Get the highest action value of the next state
    value_of_next_state = torch.max(network(next_state), dim=1)[0]
    # Calculate the temporal difference target
    td_target = reward + gamma * value_of_next_state

    # Only update target value of the chosen action
    target_values = current_values.clone()
    target_values[0, action] = td_target

    # Define optimizer and loss function
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=weight_decay)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    # Compute the loss and perform backpropagation
    loss = criterion(current_values, target_values)
    loss.backward()
    optimizer.step()


# Function to plot the results
def plot_results(total_reward_list,
                 average_reward,
                 expected_reward_shortterm_list,
                 expected_reward_longterm_list,
                 expected_reward_difference_list,
                 trial
                 ):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes[0, 0].plot(total_reward_list, color='orange')
    axes[0, 1].plot(average_reward)
    axes[1, 0].plot(expected_reward_shortterm_list, color='red', label='shortterm')
    axes[1, 0].plot(expected_reward_longterm_list, color='green', label='longterm')
    axes[1, 1].plot(expected_reward_difference_list, color='blue', label='difference')
    axes[0, 0].set_title('Cumulative Reward')
    axes[0, 0].set(xlabel='Episode', ylabel='Cumulative Reward')
    axes[0, 1].set_title('Average Reward')
    axes[0, 1].set(xlabel='Episode', ylabel='Average Reward')
    axes[1, 0].set_title('Expected Reward')
    axes[1, 0].set(xlabel='Episode', ylabel='Expected Reward')
    axes[1, 1].set_title('Expected Reward Difference')
    axes[1, 1].set(xlabel='Episode', ylabel='Expected Reward Difference')
    axes[1, 0].legend()

    if trial == 0:
        fig.suptitle('normal environment - no weight decay', fontsize=20)
    if trial == 1:
        fig.suptitle('stress environment - weight decay', fontsize=20)
    if trial == 2:
        fig.suptitle('normal environment - with weight decay', fontsize=20)

    fig.tight_layout()
    plt.show()


# Training
# Set general parameters
gamma = 0.99
epsilon = 0.3
weight_decay = 0

# Initialize network
#network = QNetwork(state_dim=100, action_dim=3)

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
    epsilon=epsilon,  # random exploration rate
    batch_size=1500,
    weight_decay=weight_decay,  # we've been using 3e-3 for depression
)


for trial in range(3):

    if trial != 1:
        # create the environment
        env = Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1, img_filename='env1.png'
            )
        )  # set render mode to "human" to see the agent moving around
    else:
        # create the environment
        env = Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1, img_filename='env5.png'
            )
        )  # set render mode to "human" to see the agent moving around


    # Initialize reward variables
    total_reward = 0
    expected_reward_longterm = 0
    expected_reward_shortterm = 0

    # Initializing lists to store the rewards
    average_reward = []
    total_reward_list = []
    expected_reward_longterm_list = []
    expected_reward_shortterm_list = []
    expected_reward_difference_list = []
    weight_decay_list = []

    steps_list = []

    # No weight decay reset for the last trail
    #if trial != 2:
     #   weight_decay = 0

    # Training loop
    for episode in range(1000):
        # Reset the environment and get the initial state
        state = env.reset()[0]

        agent.optimizer = torch.optim.Adam(
            agent.policy_net.parameters(), lr=agent.lr, weight_decay=weight_decay
        )

        # Track state-action sequence
        state_action_sequence = []

        steps = 0

        # Add a maximum of steps within an episode to be faster
        terminated = False

        while not terminated and steps < 1000:
            # Choose action using epsilon-greedy strategy
            #action = get_action(network, state, epsilon)
            action = agent.select_action(state)

            state_action_sequence.append((state, action))

            # Take the action and observe the next state and reward
            next_state, reward, terminated, _, _ = env.step(action)

            # Accumulate the total reward and memory
            total_reward += reward
            #alpha_longterm = max(0.75, 1 * 0.95)
            #expected_reward_longterm = expected_reward_longterm * alpha_longterm + reward * (1 - alpha_longterm)

            expected_reward_longterm = expected_reward_longterm * 0.95 + reward * 0.05
            expected_reward_shortterm = expected_reward_shortterm * 0.75 + reward * 0.25

            # Update the Q-network with the observed transition
            #update_network(network, state, action, reward, next_state, gamma, weight_decay)
            agent.update(
                state=state,
                new_state=next_state,
                reward=reward,
                done=terminated,
                action=action,
            )


            #weight_decay = expected_reward_difference_list[-1] * 0.01

            steps += 1

            # Move to the next state
            state = next_state

        # Update epsilon
        epsilon = max(epsilon * 0.99, 0.01)

        # Track cumulative and average rewards
        total_reward_list.append(total_reward)
        average_reward.append(total_reward / (episode + 1))
        expected_reward_longterm_list.append(expected_reward_longterm)
        expected_reward_shortterm_list.append(expected_reward_shortterm)
        expected_reward_difference_list.append(expected_reward_longterm - expected_reward_shortterm)


        weight_decay += expected_reward_difference_list[-1] * 0.01
        weight_decay = max(0, weight_decay)
        weight_decay_list.append(weight_decay)



        steps_list.append(steps)

        if episode % 100 == 0:
            print(f"episode {episode}")

    # Calculate average reward
    average_expected_reward = sum(expected_reward_difference_list) / len(expected_reward_difference_list)

    # Smooth reward lists using rolling mean
    expected_reward_longterm_list = pd.Series(expected_reward_longterm_list).rolling(window=500).mean()
    expected_reward_shortterm_list = pd.Series(expected_reward_shortterm_list).rolling(window=500).mean()
    expected_reward_difference_list = pd.Series(expected_reward_difference_list).rolling(window=500).mean()
    weight_decay_list = pd.Series(weight_decay_list).rolling(window=500).mean()

    # Print the total and average rewards
    print(f"Total reward: {total_reward}")
    print(f"Average reward: {average_reward[-1]}")
    print(f"Average Expected Reward {average_expected_reward}")
    print(f"Average steps {sum(steps_list) / len(steps_list)} ")
    if trial == 2:
        print(f"Weight decay {weight_decay}")


    # Plot the results over episodes of the total, average and expected reward
    plot_results(total_reward_list, average_reward, expected_reward_shortterm_list, expected_reward_longterm_list,
                 expected_reward_difference_list, trial)

    if trial == 1:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(weight_decay_list)
        ax.set_title('Weight Decay')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Weight Decay')



# try to adjust to get the bar that shows the percentage and current episode
#tbar = tqdm(
#        total=(len(agents) * len(envs) * N_TRIALS * iterations_per_trial), initial=0
#    )

