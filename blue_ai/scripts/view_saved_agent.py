from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH
from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
from pathlib import Path
from sys import stderr
import imageio
import argparse

filename = 'SchizophrenicAgent_0.pkl'
# filename = 'SpineLossDepression_0.pkl'
# filename = 'HealthyAgent_0.pkl'
noise_std = 0.2

_, agent, env = load_trial(DATA_PATH / filename)
env.env.render_mode = "rgb_array"

# add noise
for layer in agent.policy_net:
    if hasattr(layer, 'std'):
        print(f'changing noise std from {layer.std} to {noise_std}')
        layer.std = noise_std

# setup the environment
state, _ = env.reset()

steps_this_episode = 0
episode = 0
images = []
step = 0
plt.figure()
while 1:
    # get & execute action
    action = agent.select_action(state)
    new_state, reward, done, truncated, _ = env.step(action)

    # get image
    img = env.render()
    plt.cla()
    plt.imshow(img)
    plt.title(filename)
    plt.pause(0.001)

    # images.append(env.render())

    # use this experience to update agent
    agent.update(state, action, reward, new_state, done=False)

    # reset environment if done (ideally env would do this itself)
    if done or truncated:
        state, _ = env.reset()
        episode += 1
        if episode == 10:
            break
    else:
        state = new_state

    step += 1

# imageio.mimsave(FIGURE_PATH / f"agent_run_{file_path.name}.gif", images)
