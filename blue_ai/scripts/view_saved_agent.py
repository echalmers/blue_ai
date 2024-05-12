from blue_ai.scripts.train_agents import load_trial
import imageio

from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH

filename = DATA_PATH / "SpineLossDepression_19.pkl"
_, agent, env = load_trial(filename)
env.env.render_mode = "rgb_array"

# setup the environment
state, _ = env.reset()

steps_this_episode = 0
episode = 0
images = []
for step in range(1000):

    # get & execute action
    action = agent.select_action(state)
    new_state, reward, done, _, _ = env.step(action)

    # get image
    images.append(env.render())

    # use this experience to update agent
    agent.update(state, action, reward, new_state, done=False)

    # reset environment if done (ideally env would do this itself)
    if done or steps_this_episode > 500:
        state, _ = env.reset()
        steps_this_episode = 0
        episode += 1
        if episode == 3:
            break
    else:
        state = new_state

imageio.mimsave(FIGURE_PATH / "agent_run.gif", images)
