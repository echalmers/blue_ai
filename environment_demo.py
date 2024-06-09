#!/usr/bin/env python3

from blue_ai.envs.transient_goals import TransientGoals
import matplotlib.pyplot as plt

# instantiate environment
env = TransientGoals(
    render_mode="rgb_array", transient_reward=0.25, termination_reward=1
)
state, _ = env.reset()


def plot():
    ax.imshow(env.render())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('use arrow keys to move')
    plt.pause(0.001)


def on_key_press(event):
    if event.key == 'left':
        action = 0
    elif event.key == 'right':
        action = 1
    elif event.key == 'up':
        action = 2

    state, reward, done, _, _ = env.step(action)
    if done:
        state, _ = env.reset()
    if reward:
        print('reward:', reward)
    plot()


# create figure window
fig, ax = plt.subplots(1, 1)
fig.canvas.mpl_connect('key_press_event', on_key_press)
plot()

plt.show()