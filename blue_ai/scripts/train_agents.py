import pandas as pd
import pickle
import numpy as np
import glob
import os


def run_trial(agent, env, steps=30000, trial_id=''):

    # reset the environment
    state, _ = env.reset()

    # setup variables to track progress
    steps_this_episode = 0
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = [None] * steps

    # track agent positions to see if they get stuck
    pos = {}

    for step in range(steps):
        steps_this_episode += 1

        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        # get & execute action
        action = agent.select_action(state)
        new_state, reward, done, _, _ = env.step(action)

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if done or steps_this_episode > 500:
            print(str(agent.__class__.__name__) + f': goal reached at step {step}/{steps}' if done else '***TIMEOUT***')
            state, _ = env.reset()
            episode_num += 1
            steps_this_episode = 0
        else:
            state = new_state

        # add results to the history
        transient_goal = reward == env.unwrapped.transient_reward
        terminal_goal = reward == env.unwrapped.termination_reward
        lava = reward < 0
        stuck = max(pos.values()) > 2000
        cumulative_reward += reward
        results[step] = {
                        'trial_id': trial_id,
                        'agent': agent.__class__.__name__,
                        'step': step,
                        'episode': episode_num,
                        'reward': reward,
                        'cumulative_reward': cumulative_reward,
                        'terminal_goal': terminal_goal,
                        'transient_goal': transient_goal,
                        'lava': lava,
                        'stuck': stuck,
                    }

    results = pd.DataFrame(results)
    return results, agent, env


def save_trial(results, agent, env, filename):
    with open(filename, 'wb') as f:
        pickle.dump(
            {'results': results, 'agent': agent, 'env': env},
            f
        )


def load_trial(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['agent'], data['env']


def load_dataset(filename_patterns):
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    results = []
    for pattern in filename_patterns:
        for filename in glob.glob(os.path.join('.', 'data', pattern)):
            print(filename)
            this_result, agent, _ = load_trial(filename)
            this_result['agent'] = agent.display_name
            results.append(this_result)
    results = pd.concat(results, ignore_index=True)
    return results


if __name__ == '__main__':

    from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression, ContextDependentLearningRate, HighDiscountRate, ScaledTargets, ShiftedTargets, HighExploration
    from blue_ai.envs.transient_goals import TransientGoals
    from blue_ai.envs.custom_wrappers import Image2VecWrapper
    import os

    trial_num = 0

    for rep in range(20):

        for env in [
            Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=0.25, termination_reward=1)),
            # Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=1, termination_reward=0.25)),  # swapped reward structure
        ]:

            for agent in [
                # HealthyAgent(),
                # SpineLossDepression(),
                # ContextDependentLearningRate(),
                HighDiscountRate(),
                # ScaledTargets(),
                # HighExploration(),

                # ShiftedTargets(),
            ]:
                results, agent, env = run_trial(agent, env, steps=30_000, trial_id=trial_num)
                save_trial(
                    results, agent, env,
                    filename=os.path.join('.', 'data', f'{agent.__class__.__name__}_{"swapped_" if env.unwrapped.transient_reward > 0.25 else ""}{rep}.pkl')
                )
                trial_num += 1
