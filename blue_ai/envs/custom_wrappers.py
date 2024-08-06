import gymnasium as gym
import numpy as np


from minigrid.core.constants import OBJECT_TO_IDX, COLORS
from blue_ai.envs import custom_world_objects

object_number_map = {
    OBJECT_TO_IDX["wall"]: (0, COLORS['grey']),
    OBJECT_TO_IDX["goal"]: (1, COLORS['green']),
    OBJECT_TO_IDX["goalNoTerminate"]: (2, COLORS['blue']),
    OBJECT_TO_IDX["obstacleNoTerminate"]: (3, COLORS['red']),
}


class Image2VecWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_level=0):
        super().__init__(env)
        self.noise_level = noise_level

    def observation(self, observation):
        """
        create a new 3x7x7 state vector out of the image the env returns:
        vector[0, i, j] is 1 if the object at (i,j) is a wall
        vector[1, i, j] is 1 if the object at (i,j) is a goal
        vector[2, i, j] is 1 if the object at (i,j) is a transient goal
        vector[3, i, j] is 1 if the object at (i,j) is a hazard
        :param image: image array supplied by the TransientGoals env
        :return: a new vector as described above
        """
        image = observation["image"]
        vec = np.zeros((image.shape[0], image.shape[1], 4))
        for obj, (index, _) in object_number_map.items():
            vec[image[:, :, 0] == obj, index] = 1
        vec += np.random.normal(0, self.noise_level, size=vec.shape)
        return np.moveaxis(vec, (2, 0, 1), (0, 1, 2))

    @staticmethod
    def observation_to_image(observation, closest=False, threshold=0.25):
        """
        create an RGB image from the data provided by the observation method
        """
        rgb = np.zeros((observation.shape[1], observation.shape[2], 3))
        maxes = np.r_[observation, threshold * np.ones((1, 5, 5))].argmax(axis=0)

        for obj, (index, color) in object_number_map.items():
            slice = (maxes == index).T if closest else observation[index, :, :].T
            rgb += np.tensordot(slice, color, axes=0)
        rgb /= 256
        return rgb


if __name__ == '__main__':
    from blue_ai.envs.transient_goals import TransientGoals
    from matplotlib import pyplot as plt

    env = Image2VecWrapper(
        TransientGoals(render_mode="rgb_array", transient_reward=0.25, termination_reward=1)
    )

    state, _ = env.reset()
    state_image = env.render()
    observation_image = Image2VecWrapper.observation_to_image(state)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(state_image)
    ax[1].imshow(observation_image)
    plt.show()