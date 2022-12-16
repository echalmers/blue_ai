from typing import Optional

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.core.world_object import Key
from minigrid.core.world_object import GoalNoTerminate
from minigrid.core.world_object import KeyReward
from minigrid.core.world_object import Ball
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall, WorldObj
import numpy as np
import imageio


class TestEnv(MiniGridEnv):
    """
    <p>
        <img src="https://raw.githubusercontent.com/Farama-Foundation/Minigrid/master/figures/empty-env.png" alt="dempty-env" width="200px"/>
    </p>

    ### Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ### Mission Space

    "get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: Optional[int] = None,
        **kwargs
    ):
    
        im = imageio.imread('env1.png')
        width = len(im[0])
        height = len(im)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)


        max_steps = 4 * width**2

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        im = imageio.imread('env1.png')

        self.grid = Grid(width, height)
        obj_type = Wall
        
        hasGoal = False
        hasStart = False
        
        self.n_obstacles = 3

        
        # Generate the surrounding walls
        for x in range(0,width):
            for y in range(0,height):
                if np.sum(im[y][x]) == 255:
                    self.grid.set(x, y, obj_type())
                #goal 101 - 254
                if np.sum(im[y][x]) > 255 and np.sum(im[y][x]) < 510:
                    self.put_obj(Goal(), x, y)    
                    hasGoal = True
                #start 1 - 100
                # if np.sum(im[y][x]) > 511 and np.sum(im[y][x]) < 1020 and np.sum(im[y][x]) != 594:
                if np.sum(im[y][x]) > 511 and np.sum(im[y][x]) < 1020:    
                    # print("set start pos")
                    self.agent_start_pos = (x,y)
                    hasStart = True
                # if np.sum(im[y][x]) == 594:
                #     self.put_obj(KeyReward(), x, y)
        

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = (1,1)
            self.agent_dir = 0
        else:
            self.place_agent()

        #randomly generate
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(GoalNoTerminate())
            self.place_obj(self.obstacles[i_obst], max_tries=100)



        # Place a goal square in the bottom-right corner
        if hasGoal is False:
            self.put_obj(Goal(), width - 2, height - 2)

        # # Place the agent
        # if self.agent_start_pos is not None:
        #     self.agent_pos = self.agent_start_pos
        #     self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

        self.mission = "get to the green goal square"
