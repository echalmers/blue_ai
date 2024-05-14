from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH
from blue_ai.scripts.train_agents import load_trial
from pathlib import Path
from sys import stderr
import imageio
import argparse

parser = argparse.ArgumentParser(prog="View Saved Agent")
parser.add_argument("-a", "--agent", required=True)

args = parser.parse_args()

file_path = Path(args.agent)

if file_path.exists():
    filename = file_path
elif (DATA_PATH / file_path.name).exists():
    filename = DATA_PATH / file_path.name
else:
    print(f"Failed to open specified agent {file_path}", file=stderr)
    exit(1)

_, agent, env = load_trial(filename)
env.env.render_mode = "rgb_array"

# setup the environment
state, _ = env.reset()

steps_this_episode = 0
episode = 0
images = []
step = 0
while 1:
    # get & execute action
    action = agent.select_action(state)
    new_state, reward, done, truncated, _ = env.step(action)

    # get image
    images.append(env.render())

    # use this experience to update agent
    agent.update(state, action, reward, new_state, done=False)

    # reset environment if done (ideally env would do this itself)
    if done or truncated:
        state, _ = env.reset()
        episode += 1
        if episode == 3:
            break
    else:
        state = new_state

    step += 1

imageio.mimsave(FIGURE_PATH / f"agent_run_{file_path.name}.gif", images)
