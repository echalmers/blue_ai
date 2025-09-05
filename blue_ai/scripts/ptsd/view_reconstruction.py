from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH

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

def view_reconstruction(directory: Path ,env: Image2VecWrapper, agent_state: bool, mode: str):
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
    """

    if mode not in ['interactive', 'statistical']:
        raise ValueError("There are just two valid modes: interactive and statistical")

    with open(DATA_PATH / f'{directory.name}/interpretation_models{agent_state}.pkl', 'rb') as f:
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
            title = 'Objects reconstructed after trauma'
            subpath = "object_recon_after_trauma.png"
        case "_exposure_therapy":
            title = 'Objects reconstructed after exposure therapy'
            subpath = "object_recon_after_exposure_therapy.png"
        case "_relearned":
            title = 'Objects reconstructed after relearning'
            subpath = "object_recon_after_relearning.png"
        case "":
            title= 'Objects reconstructed before trauma'
            subpath = "object_recon_before_trauma.png"
        case _:
            title= 'Objects reconstructed at unknown point'
            subpath = "object_recon_at_unknown_point.png"
    
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
            folder_path = directory / "img"
            folder_path.mkdir(parents=True, exist_ok=True)

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
            plt.title(title, fontsize=16)
            plt.xlabel('Group', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend(title='Metrics')
            
            # Show plot
            plt.tight_layout()
            plt.savefig(folder_path/subpath)
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
        fig.suptitle(title)
        plot_interactive(state)

        plt.show()

    if mode == 'statistical':
        # create an dictionary to keep track of the number of items in the reconstructions
        recon_dict = {agent: {"# Goals": 0,  "# Transient Goals": 0, "# Hazards": 0,} for agent in interpretation_models['agent_name'].unique()}

        # the target values for important world objects
        hazard_target = torch.tensor([1.0, 0.0, 0.0])
        t_goal_target = torch.tensor([0.0, 0.0, 1.0])
        goal_target = torch.tensor([0.0, 1.0, 0.0])


        state, _ = env.reset()
        
        for _ in range(1_000):
            # place the agent in a random free position and direction
            env.unwrapped.place_agent(rand_dir = True)
            state = env.observation(env.unwrapped.gen_obs())
            state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                device=interpretation_models['agent'][0].device)

            for _, row in interpretation_models.iterrows():
                # get the reconstruction
                recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                recon[recon < 0] = 0

                # convert it to an image
                rgb = Image2VecWrapper.observation_to_image(recon.cpu() ** 1.5, closest=True)

                # update the counter inside the dictionary
                recon_dict[row['agent_name']]["# Goals"] += torch.all(rgb == goal_target, dim=-1).sum().item()
                recon_dict[row['agent_name']]["# Transient Goals"] += torch.all(rgb == t_goal_target, dim=-1).sum().item()
                recon_dict[row['agent_name']]["# Hazards"] += torch.all(rgb == hazard_target, dim=-1).sum().item()

        df = pd.DataFrame.from_dict(recon_dict, orient="index")
        print(df)
        plot_objects(df)

    
if __name__ == "__main__":

    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, 
                    transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    n_transient_obstacles=0,
                    agent_start_pos=(3,1)
                )
            )
    
    view_reconstruction(DATA_PATH / sys.argv[1], env, agent_state='', mode= 'interactive')