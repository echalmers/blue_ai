import pickle

from blue_ai.agents.dqn import SpinelossLayer
from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent, WeightDropAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH

import torch
import pandas as pd
import numpy as np
import os
def save_rehab_trial(results, agent, env, weight_dropout, weight_decay, noise_range, phase_durations,  filename, extra_info=None):
    with open(filename, "wb") as f:
        pickle.dump({
            "results": results, "agent": agent, "env": env,
             "weight_dropout": weight_dropout, "weight_decay": weight_decay,
             "noise": noise_range, "phase_durations": phase_durations, "extra_info": extra_info
        }, f)

def main():

    #choose a unique id!!
    #if there are files with the same id number already saved in your machine, these results will be
    #written into them and not saved separately!!

    id = "12"
    initial_dropout = 0.15
    weight_decay = 3e-3
    min_noise = 0.0
    max_noise = 0.0
    phase_durations = (50_000, 20_000, 20_000, 40_000)
    extra_info = "no noise"

    # List all files in the directory
    all_files = os.listdir(DATA_PATH)

    # Filter files matching the id pattern
    matching_files = [f for f in all_files if f.startswith(f"rehab_nr_{id}_trial_") and f.endswith(".pkl")]

    if matching_files:
        answer = input(
            f"Files with Id {id} already exist, proceeding may damage the existing files."
            f"Do you wish to continue? (yes/no): "
        ).strip().lower()

        if answer != "yes":
            print("Operation cancelled to avoid file corruption.")
            exit()


    for rep in range(0, 5):

        #healthy phase
        results1, agent, env = run_trial(
            trial_id=rep,
            agent=WeightDropAgent(),
            env=Image2VecWrapper(TransientGoals(render_mode="none")),
            steps=phase_durations[0],
            filename=f"{id}_healthy"
        )

        #depression on-set

        #add weight decay
        agent.optimizer = torch.optim.Adam(
            agent.policy_net.parameters(), lr=agent.lr, weight_decay=weight_decay
        )

        #add dropout
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
            # #add noise
            # for layer in agent.policy_net.children():
            #     if isinstance(layer, SpinelossLayer):
            #         layer.set_noise_scale(min_noise, max_noise)
            #         layer.add_noise()

            # remove weight decay
            agent.optimizer = torch.optim.Adam(
                agent.policy_net.parameters(), lr=agent.lr, weight_decay=0
            )

            results3, agent, env = run_trial(
                trial_id=rep,
                agent=agent,
                env=env,
                steps=phase_durations[2],
                filename=f"{id}_entropic"
            )
            results = pd.concat([results, results3], ignore_index=True)



        #rehab-phase

        #remove weight decay
        agent.optimizer = torch.optim.Adam(
            agent.policy_net.parameters(), lr=agent.lr, weight_decay=0
        )

        #remove noise
        for layer in agent.policy_net.children():
            if isinstance(layer, SpinelossLayer):
                layer.remove_noise()


        #gradually remove weight dropout
        dropout = initial_dropout
        repetitions = int(phase_durations[3] / 10_000)

        for i in range(repetitions):
            if dropout > 0:
                dropout -= 0.05
            else:
                dropout = 0.0

            print(dropout)

            for layer in agent.policy_net.children():
                if isinstance(layer, SpinelossLayer):
                    layer.set_dropout_rate(dropout)

            results3, agent, env = run_trial(
                trial_id=rep,
                agent=agent,
                env=env,
                steps=10_000,
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
            weight_decay=weight_decay,
            noise_range=(min_noise, max_noise),
            phase_durations=phase_durations,
            extra_info=extra_info,
            filename=DATA_PATH / f"rehab_nr_{id}_trial_{rep}.pkl",
            )


if __name__ == "__main__":
    main()
