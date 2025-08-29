import sys
from typing import List
from copy import deepcopy
#from pathlib import Path
#import time

from torch import nn
import matplotlib.pyplot as plt

from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, PTSDAgent, TraumaSynapticDeficitAgent
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_agents import train_agents
from blue_ai.scripts.ptsd.view_performance import view_performance
from blue_ai.scripts.ptsd.induce_traumatic_event import induce_traumatic_event
from blue_ai.scripts.ptsd.investigate_Qvalues import investigate_Qvalues
from blue_ai.scripts.ptsd.train_interpretation_models import train_interpretation_models
from blue_ai.scripts.ptsd.view_reconstruction import view_reconstruction 
from blue_ai.scripts.ptsd.post_ptsd import post_ptsd
from blue_ai.scripts.ptsd.test_performance import test_performance
from blue_ai.scripts.ptsd.view_reconstruction_loss import view_reconstruction_loss
from blue_ai.scripts.ptsd.relearning_after_trauma import relearning_after_trauma



# before using this script, get yourself familiar with the README in the ptsd folder
def main():
    # -- HYPERPARAMETERS --
    iter_per_trial = 40_000
    show_plots = True
    trauma_penalty = -100
    n_trauma_updates = 1
    n_exposure_updates = 50
    before_trauma = ''
    after_trauma = '_traumatized'
    after_relearning = '_relearned'
    after_therapy = '_exposure_therapy'
    

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
        PTSDAgent(network= network),
        TraumaSynapticDeficitAgent(network= network)
    ]
    agents_to_include = [agent.__class__.__name__ for agent in agents]

    learning_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    # hiding_penalty=True, hiding_penalty_value = 0.25
                    # see_through_walls = False IS NOT WORKING?
                )
            )
    save_env(learning_env, 'environment_during_learning', folder_path)


    trauma_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, 
                    agent_start_pos=(3,1), n_transient_obstacles=1, transient_penalty = trauma_penalty, 
                    transient_locations=[[1,4],[4,2],[5,1]], transient_obstacles=[[4,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    save_env(trauma_env, 'environment_during_trauma_experience', folder_path)


    post_trauma_env = deepcopy(trauma_env)
    post_trauma_env.unwrapped.n_transient_obstacles = 0
    post_trauma_env.unwrapped.transient_obstacles = None
    save_env(post_trauma_env, 'environment_after_trauma_experience', folder_path)


    # -- TRAIN THE AGENTS --
    train_agents(agents, learning_env, iter_per_trial, folder_path)


    # -- PLOT AND SAVE THE TRAINING PERFORMANCE --
    view_performance(folder_path, agents, "_training_in_learning_env", show_plots, '')


    # -- INDUCE THE TRAUMATIV EVENT TO ALL AGENTS --
    induce_traumatic_event(folder_path, agents_to_include, trauma_env, n_trauma_updates)


    # -- INVESTIGATE THE Q-VALUES IN THE TRAUMA ENV BUT WITHOUT THE HAZARD --
    # Das gefällt mir noch nicht so gut, wäre schöner, wenn es einfach ein plot wäre
    investigate_Qvalues(folder_path, agents_to_include, post_trauma_env, show_plots, agent_state = before_trauma)
    investigate_Qvalues(folder_path, agents_to_include, post_trauma_env, show_plots, agent_state = after_trauma)


    # -- TRAIN THE MODLES WHICH RECONSTRUCT THE VISUAL INPUT --
    train_interpretation_models(folder_path, agents_to_include, agent_state = before_trauma)
    train_interpretation_models(folder_path, agents_to_include, agent_state = after_trauma)


    # -- PLOT THE LOSS OF THE RECONSTRUCTIONS OVER THE STEPS
    view_reconstruction_loss(folder_path, agent_state=before_trauma, show_plots = show_plots)
    view_reconstruction_loss(folder_path, agent_state=after_trauma, show_plots = show_plots)


    # -- RECONSTRUCT WHAT THE AGENTS SEE --
    if show_plots:
       view_reconstruction(folder_path, post_trauma_env, agent_state = before_trauma, mode = 'interactive')
       view_reconstruction(folder_path, post_trauma_env, agent_state = after_trauma, mode = 'interactive')
    
    view_reconstruction(folder_path, post_trauma_env, agent_state = before_trauma, mode = 'statistical')
    view_reconstruction(folder_path, post_trauma_env, agent_state = after_trauma, mode = 'statistical')

    # -- RELEARNING --
    relearning_after_trauma(folder_path, agents_to_include, learning_env, iter_per_trial)
    view_performance(folder_path, agents, "_relearning_in_learning_env_after_trauma", show_plots, '_relearned')
    investigate_Qvalues(folder_path, agents_to_include, post_trauma_env, show_plots, agent_state = after_relearning)
    train_interpretation_models(folder_path, agents_to_include, agent_state = after_relearning)
    view_reconstruction_loss(folder_path, agent_state=after_relearning, show_plots = show_plots)
    if show_plots:
       view_reconstruction(folder_path, post_trauma_env, agent_state = after_relearning, mode = 'interactive')
    view_reconstruction(folder_path, post_trauma_env, agent_state = after_relearning, mode = 'statistical')


    # -- EXPOSURE THERAPY --
    post_ptsd(folder_path, agents_to_include, post_trauma_env, n_exposure_updates)

    investigate_Qvalues(folder_path, agents_to_include, post_trauma_env, show_plots, agent_state = after_therapy)
    train_interpretation_models(folder_path, agents_to_include, agent_state = after_therapy)
    view_reconstruction_loss(folder_path, agent_state=after_therapy, show_plots = show_plots)
    if show_plots:
       view_reconstruction(folder_path, post_trauma_env, agent_state = after_therapy, mode = 'interactive')
    view_reconstruction(folder_path, post_trauma_env, agent_state = after_therapy, mode = 'statistical')


    # -- TEST THE PERFORMANCE DURING THE DIFFERENT STAGES --
    test_performance(folder_path, agents_to_include, learning_env, before_trauma, 5000)
    test_performance(folder_path, agents_to_include, learning_env, after_trauma, 5000)
    test_performance(folder_path, agents_to_include, learning_env, after_relearning, 5000)
    test_performance(folder_path, agents_to_include, learning_env, after_therapy, 5000)

    view_performance(folder_path, agents, "_testing_in_learning_env_before_trauma", show_plots, '_testing')
    view_performance(folder_path, agents, "_testing_in_learning_env_after_trauma", show_plots, '_traumatized_testing')
    view_performance(folder_path, agents, "_testing_in_learning_env_after_relearning", show_plots, '_relearned_testing')
    view_performance(folder_path, agents, "_testing_in_learning_env_after_therapy", show_plots, '_exposure_therapy_testing')



def save_env(env: Image2VecWrapper, name: str, directory):
    # create the image folder, if it doesn't exist yet
    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)
    old_render_mode = env.unwrapped.render_mode
    env.unwrapped.render_mode = 'rgb_array'
    _, _ = env.reset()

    fig, ax = plt.subplots(1, 1)
    ax.axis('off')
    ax.imshow(env.render())
    ax.set_title(name)
    plt.savefig(folder_path/f'{name}.png')
    env.unwrapped.render_mode = old_render_mode


if __name__ == "__main__":
    main()