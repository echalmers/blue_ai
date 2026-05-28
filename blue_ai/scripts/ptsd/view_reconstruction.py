from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_interpretation_models import RepresentationProbe
from blue_ai.scripts.ptsd.train_agents import save_results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


import pickle
from pathlib import Path
import time
import sys

def view_reconstruction(directory: Path ,env: Image2VecWrapper, agent_state: bool, mode: str, show_plots: bool = True, seed: int = 0, reference_envs=None):
    """
    Visualize and/or statistically analyze object reconstructions from interpretation models.

    This function provides two modes of operation for examining how different agents reconstruct
    observations from the environment:

    1. Interactive mode:
        - Shows the current environment observation alongside the agent reconstructions.
        - Displays reconstructed images for each agent in real time.
        - Allows stepping through the environment using arrow keys:
        - left, right, up for movement
        - 'z' to save the current figure

    2. Statistical mode:
        - Runs multiple environment states (default 1,000) and counts occurrences of key objects
          (# Goals, # Transient Goals, # Hazards) in the reconstructions.
        - Displays a bar plot summarizing the reconstructed objects across agents.

    Args:
        directory (Path): Path to the directory containing the interpretation model pickle files 
                          and where the plots will be saved.
        env (Image2VecWrapper): Environment wrapper capable of converting observations into images.
        agent_state (str): Identifier for the agent state, used to select the pickle file and 
                           determine titles.
        mode (str): Operation mode, must be either:
                    - "interactive": live visualization with environment stepping
                    - "statistical": batch analysis with summary plots
        show_plots (bool): If True, the plots will be displayed.
        reference_envs (iterable): similarity between reconstructions and these reference envs will also be calculated
                                    (in statistical mode)
    """

    if mode not in ['interactive', 'statistical']:
        raise ValueError("There are just two valid modes: interactive and statistical")

    with open(f'{directory}/interpretation_models{agent_state}.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)
        interpretation_models['agent_name'] = interpretation_models['agent']
        interpretation_models['agent_name'] = interpretation_models['agent_name'].astype(str).replace(
            {'HealthyAgent': 'Healthy',
             'PTSDAgent': 'PTSD',
             'TraumaSynapticDeficitAgent': 'PTSDAfterTrauma'
             }
        )
    n_agents = interpretation_models['agent_name'].nunique()
    n_plots = n_agents + 2
    
    match agent_state:
        case "_traumatized":
            object_title = 'Objects reconstructed after trauma'
            object_subpath = "object_recon_after_trauma.png"
            match_title = 'Mean pixel match after trauma'
            match_subpath = 'mean_pixel_match_after_trauma.png'
        case "_exposure_therapy":
            object_title = 'Objects reconstructed after exposure therapy'
            object_subpath = "object_recon_after_exposure_therapy.png"
            match_title = 'Mean pixel match after exposure therapy'
            match_subpath = 'mean_pixel_match_after_exposure_therapy.png'
        case "_exposure_therapy_2":
            object_title = 'Objects reconstructed after exposure therapy 2'
            object_subpath = "object_recon_after_exposure_therapy_2.png"
            match_title = 'Mean pixel match after exposure therapy 2'
            match_subpath = 'mean_pixel_match_after_exposure_therapy_2.png'
        case "_relearned":
            object_title = 'Objects reconstructed after relearning'
            object_subpath = "object_recon_after_relearning.png"
            match_title = 'Mean pixel match after relearning'
            match_subpath = 'mean_pixel_match_after_relearning.png'
        case "":
            object_title= 'Objects reconstructed before trauma'
            object_subpath = "object_recon_before_trauma.png"
            match_title = 'Mean pixel match before trauma'
            match_subpath = 'mean_pixel_match_before_trauma.png'
        case "_connectivity_restoration":
            object_title= 'Objects reconstructed after restoration'
            object_subpath = "object_recon_after_connectivity_restoration.png"
            match_title = 'Mean pixel match after restoration'
            match_subpath = 'mean_pixel_match_after_connectivity_restoration.png'
        case _:
            object_title= 'Objects reconstructed at unknown point'
            object_subpath = "object_recon_at_unknown_point.png"
            match_title = 'Mean pixel match at unknown point'
            match_subpath = 'mean_pixel_match_at_unknown_point.png'
    
    def plot_interactive(state):
        for i in range(n_plots):
            ax[i].cla()

        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                device=interpretation_models['agent'][0].device)

        for index, row in interpretation_models[interpretation_models['filename'].str.contains(f'_0{agent_state}.pkl')].iterrows():
            recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
            mse = nn.MSELoss()(recon, state)
            print(row['filename'], mse)
            recon[recon < 0] = 0

            ax[2 + index].imshow(Image2VecWrapper.observation_to_image(recon.cpu() ** 1.5, closest=True))
            ax[2 + index].set_title(f"{row['agent_name']} reconstructed", fontsize = 10)  # ({round(float(mse), 2)})")

        for i in range(n_plots):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        for i in range(1, n_plots):
            t = plt.Polygon([[1.75, 4.25], [2.25, 4.25], [2, 3.75]], color='red')
            ax[i].add_patch(t)

        ax[1].set_title('visual input', fontsize = 10)
        plt.pause(0.01)
    
    def plot_objects(df: pd.DataFrame):

            # create the image folder, if it doesn't exist yet
            img_path = directory / "img"
            img_path.mkdir(parents=True, exist_ok=True)

            results_path = directory / "results"
            results_path.mkdir(parents=True, exist_ok=True)

            save_results(df, results_path / object_subpath.replace("png", "pkl"))

            # reset index so that the categories become a column
            df_reset = df.reset_index().rename(columns={'index': 'Group'})

            # melt the DataFrame to long format
            df_melted = df_reset.melt(id_vars='Group', var_name='Metric', value_name='Count')

            # assign the right colors
            colors = {
                '# Goals': 'tab:green',
                '# Transient Goals': 'tab:blue',
                '# Hazards': 'tab:red'
            }

            # Create bar plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_melted, x='Group', y='Count', hue='Metric', palette=colors)
            
            # Add titles and labels
            plt.title(object_title, fontsize=16)
            plt.xlabel('Group', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend(title='Metrics')
            
            # Show plot
            plt.tight_layout()
            plt.savefig(img_path/object_subpath)
            if show_plots:
                plt.show()
    
    def plot_matches(matches: pd.DataFrame):
        # create the image folder, if it doesn't exist yet
        folder_path = directory / "img"
        folder_path.mkdir(parents=True, exist_ok=True)

        results_path = directory / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        save_results(matches, results_path / match_subpath.replace("png", "pkl"))

        plt.figure(figsize=(5, 5))
        sns.barplot(data=matches, x='agent', y='pixel_match', hue='agent')
        plt.title(match_title, fontsize=12)
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(folder_path/match_subpath)
        if show_plots:
            plt.show()

        
    
    if mode == 'interactive':

        # make sure that the render mode is set to rgb_array
        env.unwrapped.render_mode = 'rgb_array'
        state, _ = env.reset()

        def process(event):
            folder_path = directory / "img" / "reexperiences"
            folder_path.mkdir(parents=True, exist_ok=True)

            global state
            if event.key == 'left':
                action = 0
            elif event.key == 'right':
                action = 1
            elif event.key == 'up':
                action = 2
            elif event.key == 'z':
                plt.savefig(folder_path / f'{time.time()}.png', dpi=300)
                return
            else:
                return

            state, _, done, _, _ = env.step(action)
            if done:
                state, _ = env.reset()
            plot_interactive(state)


        # create figure window
        fig, ax = plt.subplots(1, n_plots, figsize=(13, 4), constrained_layout=True)
        fig.canvas.mpl_connect('key_press_event', process)
        fig.suptitle(object_title)
        plot_interactive(state)

        plt.show()

    if mode == 'statistical':
        # create an dictionary to keep track of the number of items in the reconstructions
        recon_dict = {agent: {"# Goals": 0,  "# Transient Goals": 0, "# Hazards": 0} for agent in interpretation_models['agent_name'].unique()}
        ground_truth_dict = {"# Goals": 0,  "# Transient Goals": 0, "# Hazards": 0}
        pixel_matches = []

        # the target values for important world objects
        hazard_target = torch.tensor([1.0, 0.0, 0.0])
        t_goal_target = torch.tensor([0.0, 0.0, 1.0])
        goal_target = torch.tensor([0.0, 1.0, 0.0])

        # create a seed for reprodiction
        np.random.seed(seed)

        # place three tarnsient goals on empty spaces
        transient_locations = []
        while len(transient_locations) < 3:
            candidate = [np.random.randint(1,7),np.random.randint(1,7)]
            if candidate not in env.unwrapped.wall_locations and candidate not in [[6,6]]:
                transient_locations.append(candidate)
        env.unwrapped.transient_locations = transient_locations

        # place one hazard on and empty space
        hazard_location = []
        while len(hazard_location) < 1:
            candidate = [np.random.randint(1,7),np.random.randint(1,7)]
            if candidate not in env.unwrapped.wall_locations and candidate not in env.unwrapped.transient_locations and candidate not in [[6,6]]:
                hazard_location.append(candidate)
        env.unwrapped.transient_obstacles = hazard_location

        # get all the spaces to visit 
        places_to_visit = []
        for i in range(1,7):
            for j in range(1,7):
                places_to_visit.append((i,j))
        for i in range(len(env.unwrapped.transient_locations)):
            places_to_visit.remove(tuple(env.unwrapped.transient_locations[i]))
        for i in range(len(env.unwrapped.transient_obstacles)):
            places_to_visit.remove(tuple(env.unwrapped.transient_obstacles[i]))
        for i in range(len(env.unwrapped.wall_locations)):
            places_to_visit.remove(tuple(env.unwrapped.wall_locations[i]))
        places_to_visit.remove((6,6))

        # place the agent in every state with every direction
        for i in range(4):
            for pos in places_to_visit:
                env.unwrapped.agent_start_pos = pos
                env.unwrapped.agent_start_dir = i
                _,_ = env.reset()

                # get the state and the ground truth image  
                state = env.observation(env.unwrapped.gen_obs())
                state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                    device=interpretation_models['agent'][0].device)
                truth_image = Image2VecWrapper.observation_to_image(state[0].cpu() ** 1.5, closest=True)

                ground_truth_dict["# Goals"] += torch.all(truth_image == goal_target, dim=-1).sum().item()
                ground_truth_dict["# Transient Goals"] += torch.all(truth_image == t_goal_target, dim=-1).sum().item()
                ground_truth_dict["# Hazards"] += torch.all(truth_image == hazard_target, dim=-1).sum().item()

                for _, row in interpretation_models.iterrows():

                    match = calculate_exact_match(row['interpretation_model'], state)
                    pixel_matches.append({"agent": row['agent_name'], "pixel_match": match})

                    # get the reconstruction
                    recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                    recon[recon < 0] = 0
                    # convert it to an image
                    rgb = Image2VecWrapper.observation_to_image(recon.cpu() ** 1.5, closest=True)

                    # update the counter inside the dictionary
                    recon_dict[row['agent_name']]["# Goals"] += torch.all(rgb == goal_target, dim=-1).sum().item()
                    recon_dict[row['agent_name']]["# Transient Goals"] += torch.all(rgb == t_goal_target, dim=-1).sum().item()
                    recon_dict[row['agent_name']]["# Hazards"] += torch.all(rgb == hazard_target, dim=-1).sum().item()

        ground_truth_dict = {k: v * N_TRIALS for k, v in ground_truth_dict.items()}
        recon_dict["GroundTruth"] = ground_truth_dict
        recon_df = pd.DataFrame.from_dict(recon_dict, orient="index")
        print(recon_df)
        plot_objects(recon_df)

        pixel_matches = pd.DataFrame(pixel_matches)
        matches_per_agent = pixel_matches.groupby(['agent']).mean()
        plot_matches(matches_per_agent)

def calculate_exact_match(model, observation):
    with torch.no_grad():
        reconstruction = model.get_reconstructions(observations=observation)[1][0]
        reconstruction[reconstruction < 0] = 0
        rec_img = Image2VecWrapper.observation_to_image(reconstruction.cpu() ** 1.5, closest=True)
        obs_img = Image2VecWrapper.observation_to_image(observation[0].cpu() ** 1.5, closest=True)
        matches = np.all(rec_img == obs_img, axis=-1)
    return matches.sum() / matches.size * 100

    
if __name__ == "__main__":

    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]],#,[3,6]],
                    n_transient_obstacles=0,
                    agent_start_pos=(3,1)
                )
            )

    view_reconstruction(DATA_PATH / sys.argv[1], env, agent_state='_relearned', mode= 'interactive', seed = 0)
