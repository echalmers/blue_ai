from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_interpretation_models import RepresentationProbe
from blue_ai.scripts.ptsd.train_agents import save_results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from pathlib import Path
import sys

def view_reconstruction_loss(directory: Path, agent_state: bool, show_plots: bool):
    """
    Visualize and save the reconstruction loss and pixel match percentage of interpretation models
    for agents at a specific stage.

    This function loads precomputed interpretation model results from a pickle file, constructs
    a DataFrame containing loss and match over time for each agent, and plots two line charts:
    1. Reconstruction loss (Mean Squared Error) over environment steps.
    2. Reconstruction-observation pixel match (percentage) over environment steps.

    The plots are saved in a subdirectory "img" inside the provided directory, with a filename
    automatically selected based on the agent_state.

    Args:
        directory (Path): Path to the directory containing the interpretation model pickle files
                          and where the plots will be saved.
        agent_state (str): Identifier for the agent state, used to select the file and titles.
        show_plots (bool): If True, the plots will be displayed.
    """

    img_path = directory / "img"
    img_path.mkdir(parents=True, exist_ok=True)

    results_path = directory / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    match agent_state:
        case "_traumatized":
            loss_title = 'Loss of the interpretation model reconstructions after trauma'
            match_title = 'Match of the reconstructions and observations after trauma'
            subpath = "loss_and_match_reconstructions_after_trauma.png"
        case "_exposure_therapy":
            loss_title = 'Loss of the interpretation model reconstructions after exposure therapy'
            match_title = 'Match of the reconstructions and observations after exposure therapy'
            subpath = "loss_and_match_reconstructions_after_exposure_therapy.png"
        case "_exposure_therapy_2":
            loss_title = 'Loss of the interpretation model reconstructions after exposure therapy 2'
            match_title = 'Match of the reconstructions and observations after exposure therapy 2'
            subpath = "loss_and_match_reconstructions_after_exposure_therapy_2.png"
        case "_relearned":
            loss_title = 'Loss of the interpretation model reconstructions after relearning'
            match_title = 'Match of the reconstructions and observations after relearning'
            subpath = "loss_and_match_reconstructions_after_relearning.png"
        case "":
            loss_title= 'Loss of the interpretation model reconstructions before trauma'
            match_title = 'Match of the reconstructions and observations before trauma'
            subpath = "loss_and_match_reconstructions_before_trauma.png"
        case "_connectivity_restoration":
            loss_title= 'Loss of the interpretation model reconstructions after restoration'
            match_title = 'Match of the reconstructions and observations after restoration'
            subpath = "loss_and_match_reconstructions_after_connectivity_restoration.png"
        case _:
            loss_title= 'Loss of the interpretation model reconstructions at unknown point'
            match_title = 'Match of the reconstructions and observations at unknown point'
            subpath = "loss_and_match_reconstructions_at_unknown_point.png"

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

        for i, (loss, match) in enumerate(zip(row['losses'], row['exact_match'])):
            results.append({"agent":row['agent_name'] , "step": i , "loss":loss, "match":match})

    results = pd.DataFrame(results)
    print(results)
    save_results(results, results_path / subpath.replace("png", "pkl"))

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
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

    plt.savefig(img_path/subpath)
    if show_plots:
        plt.show()

if __name__ == "__main__":
    view_reconstruction_loss(DATA_PATH / sys.argv[1], '', True)