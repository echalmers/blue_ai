from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_interpretation_models import RepresentationProbe

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from pathlib import Path
import sys

def view_reconstruction_loss(directory: Path, agent_state: bool, show_plots: bool):

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    match agent_state:
        case "_traumatized":
            loss_title = 'Loss of the interpretation model reconstructions after trauma'
            match_title = 'Match of the reconstructions and observations after trauma'
            subpath = "loss_and_match_reconstructions_after_trauma.png"
        case "_exposure_therapy":
            loss_title = 'Loss of the interpretation model reconstructions after exposure therapy'
            match_title = 'Match of the reconstructions and observations after exposure therapy'
            subpath = "loss_and_match_reconstructions_after_exposure_therapy.png"
        case "_relearned":
            loss_title = 'Loss of the interpretation model reconstructions after relearning'
            match_title = 'Match of the reconstructions and observations after relearning'
            subpath = "loss_and_match_reconstructions_after_relearning.png"
        case _:
            loss_title= 'Loss of the interpretation model reconstructions before trauma'
            match_title = 'Match of the reconstructions and observations before trauma'
            subpath = "loss_and_match_reconstructions_before_trauma.png"

    with open(DATA_PATH / f'{directory.name}/interpretation_models{agent_state}.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)
        interpretation_models['agent_name'] = interpretation_models['agent']
        interpretation_models['agent_name'] = interpretation_models['agent_name'].astype(str).replace(
            {'HealthyAgent': 'Healthy',
             'PTSDAgent': 'PTSD',
             'TraumaSynapticDeficitAgent': 'PTSDAfterTrauma'
             }
        )
    results=[]
    for _, row in interpretation_models.iterrows():
        print(f"agent:{row['agent_name']} loss: {row['losses'][-1]}")
        print(row['exact_match'])
        for i, (loss, match) in enumerate(zip(row['losses'], row['exact_match'])):
            results.append({"agent":row['agent_name'] , "step": i , "loss":loss, "match":match})

    results = pd.DataFrame(results)
    print(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(
            data=results,
            x="step",
            y="loss",
            hue="agent",
            n_boot=1,
            ax = axes[0]
        )
    axes[0].set_title(loss_title)
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_xlabel("time (steps in environment)")
    sns.lineplot(
            data=results,
            x="step",
            y="match",
            hue="agent",
            n_boot=1,
            ax = axes[1]
        )
    axes[1].set_title(match_title)
    axes[1].set_ylabel("Match in Percent")
    axes[1].set_xlabel("time (steps in environment)")

    plt.savefig(folder_path/subpath)
    if show_plots:
        plt.show()

if __name__ == "__main__":
    view_reconstruction_loss(DATA_PATH / sys.argv[1], '', True)