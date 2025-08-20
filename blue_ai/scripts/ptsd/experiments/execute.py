import sys
from typing import List
from copy import deepcopy
#from pathlib import Path
#import time

from torch import nn
import matplotlib.pyplot as plt

from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, PTSDAgent
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_agents import train_agents
from blue_ai.scripts.ptsd.view_performance import view_performance
from blue_ai.scripts.ptsd.induce_traumatic_event import induce_traumatic_event
from blue_ai.scripts.ptsd.investigate_Qvalues import investigate_Qvalues
from blue_ai.scripts.ptsd.train_interpretation_models import train_interpretation_models
from blue_ai.scripts.ptsd.view_reconstruction import view_reconstruction 



# before using this script, get yourself familiar with the README in the ptsd folder
def main():

    # -- CREATE THE DIRECTORY --
    if len(sys.argv) < 2:
        raise ValueError("Please specify how you wanna name the directory in which the results get saved.")
    folder_path = DATA_PATH / sys.argv[1]
    folder_path.mkdir(parents=True, exist_ok=True)

    # -- AGENTS, ENVIRONMENTS & NETWORK--
    network = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(100, 25), nn.Tanh(),
        nn.Linear(25, 4)
    )
    
    agents: List[BaseAgent] = [
        HealthyAgent(network= network),
        PTSDAgent(network= network)
    ]

    learning_envs : List[Image2VecWrapper] = [Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    # hiding_penalty=True, hiding_penalty_value = 0.25
                    # see_through_walls = False IS NOT WORKING?
                )
            )
    ]
    for env in learning_envs:
        save_env(env, 'environment_during_learning', folder_path)


    trauma_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, 
                    agent_start_pos=(3,1), n_transient_obstacles=1, transient_penalty=-100, 
                    transient_locations=[[1,4],[4,2],[5,1]], transient_obstacles=[[4,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    save_env(trauma_env, 'environment_during_trauma_experience', folder_path)


    post_trauma_env = deepcopy(trauma_env)
    post_trauma_env.unwrapped.n_transient_obstacles = 0
    post_trauma_env.unwrapped.transient_obstacles = None
    save_env(post_trauma_env, 'environment_after_trauma_experience', folder_path)
    
    # -- OTHER HYPERPARAMETERS --
    iter_per_trial = 1_000
    show_plots = True

    # -- TRAIN THE AGENTS --
    train_agents(agents, learning_envs, iter_per_trial, folder_path)

    # -- PLOT AND SAVE THE PERFORMANCE --
    view_performance(folder_path, "in_learning_env", show_plots)

    # -- INDUCE THE TRAUMATIV EVENT TO ALL AGENTS --
    induce_traumatic_event(folder_path, trauma_env)

    # -- INVESTIGATE THE Q-VALUES IN THE TRAUMA ENV BUT WITHOUT THE HAZARD --
    # Das gefällt mir noch nicht so gut, wäre schöner, wenn es einfach ein plot wäre
    investigate_Qvalues(folder_path, post_trauma_env, show_plots, trauma = False,)
    investigate_Qvalues(folder_path, post_trauma_env, show_plots, trauma = True)

    # -- TRAIN THE MODLES WHICH RECONSTRUCT THE VISUAL INPUT --
    train_interpretation_models(folder_path, trauma = False)
    train_interpretation_models(folder_path, trauma = True)

    # -- RENDER THE RECONSTRUCTIONS --
    if show_plots:
        view_reconstruction(folder_path, post_trauma_env, trauma = False, mode = 'interactive')
        view_reconstruction(folder_path, post_trauma_env, trauma = True, mode = 'interactive')
    
    # -- ANALYSE THE RECONSTRUCTIONS
    view_reconstruction(folder_path, post_trauma_env, trauma = False, mode = 'statistical')
    view_reconstruction(folder_path, post_trauma_env, trauma = True, mode = 'statistical')



def save_env(env: Image2VecWrapper, name: str, directory):
    # create the image folder, if it doesn't exist yet
    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    env.unwrapped.render_mode = 'rgb_array'
    _, _ = env.reset()

    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    ax.imshow(env.render())
    ax.set_title(name)
    plt.savefig(folder_path/f'{name}.png')



if __name__ == "__main__":
    main()