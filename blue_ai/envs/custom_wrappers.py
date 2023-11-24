import gymnasium as gym
import numpy as np


from minigrid.core.constants import OBJECT_TO_IDX
object_vector_map = {
    OBJECT_TO_IDX['wall']: [1, 0, 0, 0],
    OBJECT_TO_IDX['goal']: [0, 1, 0, 0],
    OBJECT_TO_IDX['goalNoTerminate']: [0, 0, 1, 0],
    OBJECT_TO_IDX['obstacleNoTerminate']: [0, 0, 0, 1],
}


class AbsolutePositionWrapper(gym.ObservationWrapper):

    def observation(self, observation):
        return (observation['direction'], *observation['position'])


class Image2VecWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        """
        create a new 3x7x7 state vector out of the image the env returns:
        vector[i, j, 0] is 1 if the object at (i,j) is a wall
        vector[i, j, 1] is 1 if the object at (i,j) is a goal
        vector[i, j, 2] is 1 if the object at (i,j) is a transient goal
        :param image: image array supplied by the TransientGoals env
        :return: a new vector as described above
        """
        image = observation['image']
        vec = np.zeros((image.shape[0], image.shape[1], 4))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                vec[i, j, :] = object_vector_map.get(image[i, j, 0], [0, 0, 0, 0])
        return np.moveaxis(vec, (2, 0, 1), (0, 1, 2))