from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_interpretation_models import RepresentationProbe
from blue_ai.scripts.ptsd.train_agents import save_results
from blue_ai.scripts.ptsd.view_reconstruction import view_reconstruction
import pickle
import pickle
from pathlib import Path
import time
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine as cosine_similarity


def view_reconstruction(directory: Path, test_env: Image2VecWrapper, trauma_env: Image2VecWrapper, agent_state: str = None,
                        seed: int = 0):
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
        test_env (Image2VecWrapper): Environment wrapper capable of converting observations into images.
        trauma_env (Image2VecWrapper): Trauma environment
        agent_state (str): Identifier for the agent state, used to select the pickle file and
                           determine titles.
        seed (int): random seed for setting up test env
    """

    # get trauma state for reference
    trauma_env.reset()
    trauma_state = trauma_env.observation(trauma_env.unwrapped.gen_obs())


    with open(f'{directory}/interpretation_models{agent_state}.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)
        interpretation_models['agent_name'] = interpretation_models['agent']
        interpretation_models['agent_name'] = interpretation_models['agent_name'].astype(str).replace(
            {'HealthyAgent': 'Healthy',
             'PTSDAgent': 'PTSD',
             'TraumaSynapticDeficitAgent': 'PTSDAfterTrauma'
             }
        )

    # create an dictionary to keep track of the number of items in the reconstructions
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
        candidate = [np.random.randint(1, 7), np.random.randint(1, 7)]
        if candidate not in test_env.unwrapped.wall_locations and candidate not in [[6, 6]]:
            transient_locations.append(candidate)
    test_env.unwrapped.transient_locations = transient_locations

    # place one hazard on and empty space
    hazard_location = []
    while len(hazard_location) < 1:
        candidate = [np.random.randint(1, 7), np.random.randint(1, 7)]
        if candidate not in test_env.unwrapped.wall_locations and candidate not in test_env.unwrapped.transient_locations and candidate not in [
            [6, 6]]:
            hazard_location.append(candidate)
    test_env.unwrapped.transient_obstacles = hazard_location

    # get all the spaces to visit
    places_to_visit = []
    for i in range(1, 7):
        for j in range(1, 7):
            places_to_visit.append((i, j))
    for i in range(len(test_env.unwrapped.transient_locations)):
        places_to_visit.remove(tuple(test_env.unwrapped.transient_locations[i]))
    for i in range(len(test_env.unwrapped.transient_obstacles)):
        places_to_visit.remove(tuple(test_env.unwrapped.transient_obstacles[i]))
    for i in range(len(test_env.unwrapped.wall_locations)):
        places_to_visit.remove(tuple(test_env.unwrapped.wall_locations[i]))
    places_to_visit.remove((6, 6))


    # place the agent in every state with every direction
    for i in range(4):
        for pos in places_to_visit:
            test_env.unwrapped.agent_start_pos = pos
            test_env.unwrapped.agent_start_dir = i
            _, _ = test_env.reset()

            # get the state and the ground truth image
            state = test_env.observation(test_env.unwrapped.gen_obs())
            state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                 device=interpretation_models['agent'][0].device)
            truth_image = Image2VecWrapper.observation_to_image(state[0].cpu() ** 1.5, closest=True)

            for _, row in interpretation_models.iterrows():
                # get the reconstruction
                with torch.no_grad():
                    recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                    recon[recon < 0] = 0

                # measure similarity of reconstruction with trauma state
                blurred_trauma_state = gaussian_filter(trauma_state, sigma=0.5)
                blurred_recon = gaussian_filter(recon, sigma=0.5)

                # plt.subplot(1, 2, 1)
                # plt.imshow(Image2VecWrapper.observation_to_image(trauma_state, closest=False))
                # plt.subplot(1, 2, 2)
                # plt.imshow(Image2VecWrapper.observation_to_image(blurred_trauma_state, closest=False))
                # plt.show()


                # mse = float(((blurred_recon - blurred_trauma_state) ** 2).mean())
                # mse = float(((recon - trauma_state) ** 2).mean())
                # pixel_matches.append({"agent": row['agent_name'], "trauma_state_mse": mse})

                recon_img = Image2VecWrapper.observation_to_image(recon ** 1.5, closest=True)
                trauma_img = Image2VecWrapper.observation_to_image(trauma_state ** 1.5, closest=True)
                # plt.subplot(1, 2, 1)
                # plt.imshow(recon_img)
                # plt.subplot(1, 2, 2)
                # plt.imshow(trauma_img)
                # plt.pause(0.1)
                # plt.cla()

                # matches = np.all(recon_img == trauma_img, axis=-1)
                # matches = matches.sum() / matches.size * 100

                mse = float(((recon_img - trauma_img) ** 2).mean())

                cos_sim = 1 - cosine_similarity(recon.flatten(), trauma_state.flatten())

                pixel_matches.append({"agent": row['agent_name'], "agent_state": agent_state, "sim": cos_sim})

    return pixel_matches





def calculate_exact_match(model, observation, reference_observation):
    with torch.no_grad():
        reconstruction = model.get_reconstructions(observations=observation)[1][0]
        reconstruction[reconstruction < 0] = 0
        rec_img = Image2VecWrapper.observation_to_image(reconstruction.cpu() ** 1.5, closest=True)
        obs_img = Image2VecWrapper.observation_to_image(observation[0].cpu() ** 1.5, closest=True)
        matches = np.all(rec_img == obs_img, axis=-1)
    return matches.sum() / matches.size * 100



if __name__ == '__main__':

    test_env = Image2VecWrapper(
        TransientGoals(
            render_mode="none", transient_reward=0.25, termination_reward=1,
            transient_locations=[[1, 4], [4, 2], [5, 1]],
            wall_locations=[[3, 2], [3, 3], [3, 4], [3, 5]],  # ,[3,6]],
            n_transient_obstacles=0,
            agent_start_pos=(3, 1)
        )
    )

    traumatic_env = Image2VecWrapper(
        TransientGoals(
            render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3 ,1),
            n_transient_obstacles=1, transient_penalty=-1000, transient_locations=[[1 ,4] ,[4 ,2] ,[5 ,1]], transient_obstacles=[[4 ,1]],
            wall_locations =[[3 ,2] ,[3 ,3] ,[3 ,4] ,[3 ,5]], env_name='trauma_env'
        )
    )

    data = []
    for seed in range(1):
        # data.extend(view_reconstruction(DATA_PATH, test_env, traumatic_env, agent_state='', seed=seed))
        data.extend(view_reconstruction(DATA_PATH, test_env, traumatic_env, agent_state='_traumatized', seed=seed))
        data.extend(view_reconstruction(DATA_PATH, test_env, traumatic_env, agent_state='_relearned', seed=seed))
    df = pd.DataFrame(data)
    df['agent_state'] = df['agent_state'].replace({'': 'before trauma', '_traumatized': 'immediately\nafter trauma', '_relearned': 'after relearning'})
    df['agent'] = df['agent'].replace({
        'Healthy': 'healthy', 'PTSD': 'pre-existing deficits', 'PTSDAfterTrauma': 'post-trauma deficits'
    })

    sns.lineplot(df, x='agent_state', y='sim', palette=['grey', 'royalblue', 'midnightblue'], hue='agent', errorbar='se')
    plt.xlabel('')
    plt.legend().set_title("")
    plt.title('similarity between agents\' perceptions and the trauma state')
    plt.ylabel('cosine similarity')


    plt.show()
