import pickle
from dqn import DQN, image_to_vec, TransitionMemory
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    dropout = 70

    # load agent
    with open(f'dqn_do{dropout}.pkl', 'rb') as f:
        agent = pickle.load(f)

    # start the environment
    env = gym.make('blue_ai_envs/TransientGoals', tile_size=64, render_mode='human')
    state = env.reset()
    state = image_to_vec(state[0]['image'])

    while True:
        # get & execute action chosen by DQN agent
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)
        plt.title('deep Q network' if dropout == 0 else 'simulated depression')
        new_state = image_to_vec(new_state['image'])

        # reset environment if done (ideally env would do this itself)
        if done:
            state = env.reset()
            state = image_to_vec(state[0]['image'])
        else:
            state = new_state