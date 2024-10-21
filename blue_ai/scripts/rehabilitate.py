import pickle

from blue_ai.agents.dqn import SpinelossLayer
from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent, WeightDropAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH


import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import polars as pl
from tqdm import trange

def save_rehab_trial(results, agent, env, weight_dropout, noise_range, phase_durations,  filename):
    with open(filename, "wb") as f:
        pickle.dump(
            {"results": results, "agent": agent, "env": env,
             "weight_dropout": weight_dropout, "noise": noise_range, "phase_durations": phase_durations}, f
        )

def main():

    id = "4"
    initial_dropout = 0.1
    min_noise = 0.8
    max_noise = 1.5
    phase_durations = (30_000, 20_000, 15_000, 40_000)

    for rep in range(5, 10):

        #healthy phase
        results1, agent, env = run_trial(
            trial_id=rep,
            agent=WeightDropAgent(),
            env=Image2VecWrapper(TransientGoals(render_mode="none")),
            steps=phase_durations[0],
            filename=f"{id}_healthy"
        )


        #depression on-set
        for layer in agent.policy_net.children():
            if isinstance(layer, SpinelossLayer):
                layer.set_dropout_rate(initial_dropout)

        results2, agent, env = run_trial(
            trial_id=rep,
            agent=agent,
            env=env,
            steps=phase_durations[1],
            filename=f"{id}_depressed"

        )

        results = pd.concat([results1, results2], ignore_index=True)


        # high-entropy phase
        if phase_durations[2] > 0:
            for layer in agent.policy_net.children():
                if isinstance(layer, SpinelossLayer):
                    layer.set_noise_scale(min_noise, max_noise)
                    layer.add_noise()


            results3, agent, env = run_trial(
                trial_id=rep,
                agent=agent,
                env=env,
                steps=phase_durations[2],
                filename=f"{id}_entropic"
            )
            results = pd.concat([results, results3], ignore_index=True)


        # #rehab-phase
        # dropout = initial_dropout
        # repetitions = int(phase_durations[3]/10_000)
        repetitions = 1
        dropout = 0
        for i in range(repetitions):
            if dropout > 0:
                dropout -= 0.1
            else:
                dropout = 0.0

            print(dropout)

            for layer in agent.policy_net.children():
                if isinstance(layer, SpinelossLayer):
                    layer.set_dropout_rate(dropout)
                    layer.set_noise_scale(min_noise, max_noise)
                    layer.add_noise()

            results3, agent, env = run_trial(
                trial_id=rep,
                agent=agent,
                env=env,
                steps=phase_durations[3],
                filename=f"{id}_treated"
            )
            results = pd.concat([results, results3], ignore_index=True)

        results["cumulative_reward"] = results["reward"].cumsum()
        results['rolling_avg_reward'] = results['reward'].rolling(window=8000).mean()
        results["step"] = np.arange(results.shape[0])


        save_rehab_trial(
            results,
            agent,
            env,
            weight_dropout=initial_dropout,
            noise_range=(min_noise, max_noise),
            phase_durations=phase_durations,
            filename=DATA_PATH / f"rehab_nr_{id}_trial_{rep}.pkl",
            )


if __name__ == "__main__":
    main()
