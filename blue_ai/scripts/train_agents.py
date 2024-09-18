from typing import Any, Dict, List, Tuple, TypedDict
import pandas as pd
import pickle
from copy import deepcopy

import torch.nn
import polars as pl
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from tqdm import tqdm

from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.agents.agent_classes import *

activations = {}

def create_hook(name, record_activations_flag, step_idx):
    def hook(module, input, output):

        if record_activations_flag[0]:
            if name not in activations:
                activations[name] = []  # Initialize list for this layer

            # Store activation data in a structured way
            step = step_idx[0]
            output_np = output.detach().cpu().numpy()
            layer_activations = pl.DataFrame({
                "step": [step] * output_np.shape[0],
                "layer": [name] * output_np.shape[0],
                "activation": output_np.tolist()
            })
            activations[name].append(layer_activations)
    return hook


def save_activations_to_parquet(filename):
    if activations:
        activation_dfs = []
        for layer_name, dfs in activations.items():
            activation_dfs.extend(dfs)

        # Concatenate all DataFrames into a single DataFrame
        activation_df = pl.concat(activation_dfs)

        # Save to Parquet using the 'snappy' compression
        activation_df.write_parquet(filename, compression='gzip')
        activations.clear()
        print(f"Activations successfully saved to {filename}")
    else:
        print("No activations to save.")


def run_trial(agent: BaseAgent, env, steps=30, trial_id="", tbar=None, filename=None):
    state, _ = env.reset()
    # setup variables to track progress
    steps_this_episode = 0
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = [None] * steps

    # Setup for storing activations
    hooks = []
    record_activations_flag = [False]
    step_idx = [0]

    for name, layer in agent.policy_net.named_modules():
        if isinstance(layer, torch.nn.Linear):
            hook = layer.register_forward_hook(create_hook(name, record_activations_flag, step_idx))
            hooks.append(hook)

    # track agent positions to see if they get stuck
    pos: Dict[Tuple[int, int], int] = {}
    if tbar is not None:
        tbar.set_postfix(
            agent=agent.__class__.__name__, env=env.__class__.__name__, trial=trial_id
        )

    for step in range(steps):
        steps_this_episode += 1
        step_idx[0] = step

        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        # get & execute action
        record_activations_flag[0] = True
        action = agent.select_action(state)
        record_activations_flag[0] = False

        new_state, reward, done, truncated, _ = env.step(action)
        # use this experience to update agent
        loss = agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if truncated or done:
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
            "trial_id": trial_id,
            "agent": agent.__class__.__name__,
            "step": step,
            "episode": episode_num,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "terminal_goal": terminal_goal,
            "transient_goal": transient_goal,
            "lava": lava,
            "stuck": stuck,
            "mean_synapse": next(agent.policy_net.parameters()).mean().item(),
            "num_pos_synapse": (next(agent.policy_net.parameters()) > 0).sum().item(),

        }

        if tbar is not None:
            tbar.update()

    results = pd.DataFrame(results)

    # Save activations periodically
    activation_filename = DATA_PATH / f"{filename or agent.__class__.__name__}_{trial_id}_activations.parquet"
    save_activations_to_parquet(activation_filename)

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    return results, agent, env


def save_trial(results, agent, env, filename):
    with open(filename, "wb") as f:
        pickle.dump({"results": results, "agent": agent, "env": env}, f)


def load_trial(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["results"], data["agent"], data["env"]


def load_dataset(filename_patterns):
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    results = []
    for pattern in filename_patterns:
        files = list(DATA_PATH.glob(pattern))
        for filename in tqdm(files, leave=False, total=len(files)):
            this_result, agent, _ = load_trial(filename)
            this_result["agent"] = (
                agent.display_name
                if hasattr(agent, "display_name")
                else agent.__class__.__name__
            )

            this_result["filename"] = filename
            results.append(this_result)
    results = pd.concat(results, ignore_index=True)
    return results


def trial(agent: BaseAgent, env, rep, trial_num, tbar=None, steps=30_000):
    results, agent, env = run_trial(
        agent,
        env,
        steps=steps,
        trial_id=trial_num,
        tbar=tbar,
    )

    filename = (
            DATA_PATH
            / f'{agent.file_display_name()}_{"swapped_" if env.unwrapped.transient_reward > 0.25 else ""}{rep}.pkl'
    )

    save_trial(results, agent, env, filename)
    return trial_num


def main():
    iterations_per_trial = 30_000
    trial_num = 0

    agents: List[BaseAgent] = [
        HealthyAgent(),
        SpineLossDepression(),
        # ContextDependentLearningRate(),
        # HighDiscountRate(),
        # ScaledTargets(),
        # HighExploration(),
        # ShiftedTargets(),
        # PositiveLossAgent(),
        # ReluActivation(),
        # ReluLossActivation(),
    ]
    envs = [
        Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1
            )
        ),
        # swapped reward structure
        # Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=1, termination_reward=0.25)),
    ]

    # # Setup agent sweep
    # agents += [
    #     PositiveLossAgent(alpha=(2**-x), embed_alpha_in_filename=True)
    #     for x in range(1, 6)
    # ]

    tbar = tqdm(
        total=(len(agents) * len(envs) * N_TRIALS * iterations_per_trial), initial=0
    )

    for rep in range(N_TRIALS):
        for env in envs:
            for agent in agents:
                tbar.set_postfix(
                    agent=agent.__class__.__name__, env=env.__class__.__name__, rep=rep
                )
                trial(
                    deepcopy(agent),
                    env,
                    rep,
                    trial_num,
                    tbar=tbar,
                    steps=iterations_per_trial,
                )
                trial_num += 1
    print(len(activations))


if __name__ == "__main__":
    main()
