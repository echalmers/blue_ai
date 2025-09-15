import sys
from typing import List
from copy import deepcopy

from torch import nn
import matplotlib.pyplot as plt

from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, PTSDAgent, TraumaSynapticDeficitAgent
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH

from blue_ai.scripts.ptsd.post_ptsd_2 import post_ptsd_2
from blue_ai.scripts.ptsd.post_ptsd_3 import post_ptsd_3
from blue_ai.scripts.ptsd.train_agents import train_agents
from blue_ai.scripts.ptsd.view_performance import view_performance
from blue_ai.scripts.ptsd.test_performance import test_performance
from blue_ai.scripts.ptsd.view_reconstruction import view_reconstruction 
from blue_ai.scripts.ptsd.investigate_Qvalues import investigate_Qvalues
from blue_ai.scripts.ptsd.view_position_heatmap import view_position_heatmap
from blue_ai.scripts.ptsd.induce_traumatic_event import induce_traumatic_event
from blue_ai.scripts.ptsd.relearning_after_trauma import relearning_after_trauma
from blue_ai.scripts.ptsd.view_Qvalue_differences import view_Qvalue_differences
from blue_ai.scripts.ptsd.view_reconstruction_loss import view_reconstruction_loss
from blue_ai.scripts.ptsd.train_interpretation_models import train_interpretation_models



# before using this script, get yourself familiar with the README in the ptsd folder
def main():
    # -- HYPERPARAMETERS --
    iter_per_trial = 40_000
    show_plots = True
    trauma_penalty = -1000
    n_trauma_updates = 1
    before_trauma = ''
    after_trauma = '_traumatized'
    after_relearning = '_relearned'
    after_therapy_2 = '_exposure_therapy_2'
    after_restoration = '_connectivity_restoration'
    

    # -- CREATE THE DIRECTORY --
    if len(sys.argv) < 2:
        raise ValueError("Please specify how you wanna name the directory in which the results get saved.")
    folder_path = DATA_PATH / sys.argv[1]
    folder_path.mkdir(parents=True, exist_ok=True)


    # -- AGENTS, ENVIRONMENTS & NETWORK--
    network = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(100, 25), nn.Tanh(),
        nn.Linear(25, 3)
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
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]]#,[3,6]],
                )
            )


    trauma_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, 
                    agent_start_pos=(3,1), n_transient_obstacles=1, transient_penalty = trauma_penalty, 
                    transient_locations=[[1,4],[4,2],[5,1]], transient_obstacles=[[4,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]]#,[3,6]]
                )
            )


    post_trauma_env = deepcopy(trauma_env)
    post_trauma_env.unwrapped.n_transient_obstacles = 0
    post_trauma_env.unwrapped.transient_obstacles = None


    therapy_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 0,
                    transient_locations=[[4,1],[3,1],[2,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]]#,[3,6]],
                )
            )
    
    # save all the different environments
    save_env(learning_env, 'environment_during_learning', folder_path)
    save_env(trauma_env, 'environment_during_trauma_experience', folder_path)
    save_env(post_trauma_env, 'environment_after_trauma_experience', folder_path)
    save_env(therapy_env, 'environment_during_therapy', folder_path)



    # -- BEFORE TRAUMA --
    # train the agents
    train_agents(agents, learning_env, iter_per_trial, folder_path)

    # plot the training performance
    view_performance(folder_path, agents, "_training_in_learning_env", show_plots, '')
    # investigate the qvalues in trauma. environment but without the hazard
    investigate_Qvalues(folder_path, agents_to_include, deepcopy(post_trauma_env), show_plots, before_trauma)
    # train the reconstruction models
    train_interpretation_models(folder_path, agents_to_include, before_trauma)
    # plot the loss of the recosntruction models over time
    view_reconstruction_loss(folder_path, before_trauma, show_plots)
    # reconstruct what the agent sees
    if show_plots:
       view_reconstruction(folder_path, deepcopy(post_trauma_env), before_trauma, 'interactive')
    view_reconstruction(folder_path, deepcopy(learning_env), before_trauma, 'statistical', show_plots)
    # test the performance after the learning
    test_performance(folder_path, agents_to_include, deepcopy(learning_env), before_trauma, 5000)
    # plot the performance of the testing
    view_performance(folder_path, agents, "_testing_in_learning_env_before_trauma", show_plots, '_testing')
    # create a heatmap showing the position of the agents during testing
    view_position_heatmap(folder_path, agents_to_include, show_plots, '_testing')
    # create a plot and a heatmap commparing qvalues at different positions
    view_Qvalue_differences(folder_path, agents_to_include, before_trauma, deepcopy(post_trauma_env), show_plots)



    # -- TRAUMA --
    # induce the traumatic event to all agents
    induce_traumatic_event(folder_path, agents_to_include, deepcopy(trauma_env), n_trauma_updates)

    investigate_Qvalues(folder_path, agents_to_include, deepcopy(post_trauma_env), show_plots, after_trauma)
    train_interpretation_models(folder_path, agents_to_include, after_trauma)
    view_reconstruction_loss(folder_path, after_trauma, show_plots)
    if show_plots:
       view_reconstruction(folder_path, deepcopy(post_trauma_env), after_trauma, 'interactive')
    view_reconstruction(folder_path, deepcopy(learning_env), after_trauma, 'statistical', show_plots)
    test_performance(folder_path, agents_to_include, deepcopy(learning_env), after_trauma, 5000)
    view_performance(folder_path, agents, "_testing_in_learning_env_after_trauma", show_plots, '_traumatized_testing')
    view_position_heatmap(folder_path, agents_to_include, show_plots, '_traumatized_testing')
    view_Qvalue_differences(folder_path, agents_to_include, after_trauma, deepcopy(post_trauma_env), show_plots)



    # -- RELEARNING --
    relearning_after_trauma(folder_path, agents_to_include, deepcopy(learning_env), iter_per_trial)

    view_performance(folder_path, agents, "_relearning_in_learning_env_after_trauma", show_plots, '_relearned')
    investigate_Qvalues(folder_path, agents_to_include, deepcopy(post_trauma_env), show_plots, after_relearning)
    train_interpretation_models(folder_path, agents_to_include, after_relearning)
    view_reconstruction_loss(folder_path, after_relearning, show_plots)
    if show_plots:
       view_reconstruction(folder_path, deepcopy(post_trauma_env), after_relearning, 'interactive')
    view_reconstruction(folder_path, deepcopy(learning_env), after_relearning, 'statistical', show_plots)
    test_performance(folder_path, agents_to_include, deepcopy(learning_env), after_relearning, 5000)
    view_performance(folder_path, agents, "_testing_in_learning_env_after_relearning", show_plots, '_relearned_testing')
    view_position_heatmap(folder_path, agents_to_include, show_plots, '_relearned_testing')
    view_Qvalue_differences(folder_path, agents_to_include, after_relearning, deepcopy(post_trauma_env), show_plots)



    # -- EXPOSURE THERAPY v 2.0--
    post_ptsd_2(folder_path, agents_to_include, deepcopy(therapy_env))

    investigate_Qvalues(folder_path, agents_to_include, deepcopy(post_trauma_env), show_plots, after_therapy_2)
    train_interpretation_models(folder_path, agents_to_include, after_therapy_2)
    view_reconstruction_loss(folder_path, after_therapy_2, show_plots)
    if show_plots:
       view_reconstruction(folder_path, deepcopy(post_trauma_env), after_therapy_2, 'interactive')
    view_reconstruction(folder_path, deepcopy(learning_env), after_therapy_2, 'statistical', show_plots)
    test_performance(folder_path, agents_to_include, deepcopy(learning_env), after_therapy_2, 5000)
    view_performance(folder_path, agents, "_testing_in_learning_env_after_therapy_2", show_plots, '_exposure_therapy_2_testing')
    view_position_heatmap(folder_path, agents_to_include, show_plots, '_exposure_therapy_2_testing')
    view_Qvalue_differences(folder_path, agents_to_include, after_therapy_2, deepcopy(post_trauma_env), show_plots)



    # -- CONNECTIVITY RESTORATION --
    post_ptsd_3(folder_path, agents_to_include, deepcopy(learning_env), iter_per_trial)

    investigate_Qvalues(folder_path, agents_to_include, deepcopy(post_trauma_env), show_plots, after_restoration)
    train_interpretation_models(folder_path, agents_to_include, after_restoration)
    view_reconstruction_loss(folder_path, after_restoration, show_plots)
    if show_plots:
       view_reconstruction(folder_path, deepcopy(post_trauma_env), after_restoration, 'interactive')
    view_reconstruction(folder_path, deepcopy(learning_env), after_restoration, 'statistical', show_plots)
    test_performance(folder_path, agents_to_include, deepcopy(learning_env), after_restoration, 5000)
    view_performance(folder_path, agents, "_testing_in_learning_env_after_connectivity_restoration", show_plots, '_connectivity_restoration_testing')
    view_position_heatmap(folder_path, agents_to_include, show_plots, '_connectivity_restoration_testing')
    view_Qvalue_differences(folder_path, agents_to_include, after_restoration, deepcopy(post_trauma_env), show_plots)



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