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
            title = 'Loss of the interpretation model reconstructions after trauma'
            subpath = "loss_reconstructions_after_trauma.png"
        case "_exposure_therapy":
            title = 'Loss of the interpretation model reconstructions after exposure therapy'
            subpath = "loss_reconstructions_after_exposure_therapy.png"
        case "_relearned":
            title = 'Loss of the interpretation model reconstructions after relearning'
            subpath = "loss_reconstructions_after_relearning.png"
        case _:
            title= 'Loss of the interpretation model reconstructions before trauma'
            subpath = "loss_reconstructions_before_trauma.png"

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
        for i, loss in enumerate(row['losses']):
            results.append({"agent":row['agent_name'] , "step": i , "loss":loss })

    results = pd.DataFrame(results)
    print(results)
    sns.lineplot(
            data=results,
            x="step",
            y="loss",
            hue="agent",
            n_boot=1,
        )
    plt.title(title)
    plt.ylabel("Loss (MSE)")
    plt.xlabel("time (steps in environment)")
    plt.savefig(folder_path/subpath)
    if show_plots:
        plt.show()

if __name__ == "__main__":
    view_reconstruction_loss(DATA_PATH / sys.argv[1], '_traumatized', True)