from pathlib import Path
from sys import stderr
from minigrid.core.grid import Grid, WorldObj
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from blue_ai.envs.custom_world_objects import GoalNoTerminate
from blue_ai.envs.custom_world_objects import ObstacleNoTerminate
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall
import numpy as np
import imageio
from enum import IntEnum

from dataclasses import dataclass


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2

    # Done completing task
    done = 3


@dataclass
class TransientGoals(MiniGridEnv):
    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        termination_reward=1,
        transient_reward=0.25,
        n_transient_goals=3,
        img_filename="env1.png",
        transient_locations=None,
        replace_transient_goals=False,
        transient_penalty=-1,
        n_transient_obstacles=1,
        transient_obstacles=None,
        replace_transient_obstacles=False,
        **kwargs,
    ):

        # In order to prevent patthing issues we need to ensure that we path
        # relative to the location of this file rather than the run location
        image_path = Path(__file__).parent / img_filename
        self.im = imageio.imread(image_path)

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.termination_reward = termination_reward
        self.transient_reward = transient_reward
        self.n_transient_goals = n_transient_goals
        self.transient_locations = transient_locations
        self.replace_transient_goals = replace_transient_goals
        self.transient_penalty = transient_penalty
        self.n_transient_obstacles = n_transient_obstacles
        self.transient_obstacles = transient_obstacles
        self.replace_transient_obstacles = replace_transient_obstacles

        mission_space = MissionSpace(mission_func=self._gen_mission)

        ## This seems really arbitrary
        max_steps = 4 * (len(self.im[0])) ** 2

        super().__init__(
            mission_space=mission_space,
            width=len(self.im[0]),
            height=len(self.im),
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            agent_view_size=5,
            **kwargs,
        )
        self.actions = Actions

    def _replace_transient_obstacles(self, reward):
        self.penalties.append(ObstacleNoTerminate(reward=reward))
        self.place_obj(self.penalties[-1], max_tries=100)

    """
    Turns the agent to the left
    """

    def _turn_left(self):
        self.agent_dir = (self.agent_dir - 1) % 4

    """
    Turns the agent to the right
    """

    def _turn_right(self):
        self.agent_dir = (self.agent_dir + 1) % 4

    """
    Determines if the current state is a end state
    """

    def _check_termination_conditions(self, cell_type: str):
        return cell_type == "goal" or cell_type == "lava"

    def _get_reward_amount(
        self, fwd_cell: GoalNoTerminate | ObstacleNoTerminate | WorldObj
    ):
        kind = fwd_cell.type
        reward = 0.0

        if hasattr(fwd_cell, "reward"):
            reward = fwd_cell.reward

        if kind == "goal":
            reward = self.termination_reward

        return reward

    def _handle_forward(self):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        terminated, reward = False, 0

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)

        if fwd_cell is None:
            return terminated, reward

        terminated = self._check_termination_conditions(fwd_cell.type)
        reward = self._get_reward_amount(fwd_cell)

        is_goal_no_terminate = fwd_cell.type == "goalNoTerminate"
        is_obstacle_no_terminate = fwd_cell.type == "obstacleNoTerminate"

        # Create a new obstacle/reward if the current one is "used"

        if self.replace_transient_goals and is_goal_no_terminate:
            self.obstacles.append(GoalNoTerminate(self.transient_reward))
            self.place_obj(self.obstacles[-1], max_tries=100)

        if self.replace_transient_obstacles and is_obstacle_no_terminate:
            self.penalties.append(ObstacleNoTerminate(self.transient_penalty))
            self.place_obj(self.penalties[-1], max_tries=100)

        # Remove game object once traversed
        if is_obstacle_no_terminate or is_goal_no_terminate:
            self.grid.set(fwd_pos[0], fwd_pos[1], None)

        return terminated, reward

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False

        match action:
            case self.actions.left:
                self._turn_left()
            case self.actions.right:
                self._turn_right()
            case self.actions.forward:
                terminated, reward = self._handle_forward()

        terminated |= action == self.actions.done
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        hasGoal = False

        for x in range(0, width):
            for y in range(0, height):
                if np.sum((self.im)[y][x]) == 255:
                    self.grid.set(x, y, Wall())
                if np.sum((self.im)[y][x]) > 255 and np.sum((self.im)[y][x]) < 510:
                    self.put_obj(Goal(), x, y)
                    hasGoal = True

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # add transient goals
        self.obstacles = []
        if self.transient_locations is not None:
            for location in self.transient_locations:
                self.obstacles.append(GoalNoTerminate(reward=self.transient_reward))
                self.grid.set(location[0], location[1], self.obstacles[0])
        else:
            for i_obst in range(self.n_transient_goals):
                self.obstacles.append(GoalNoTerminate(reward=self.transient_reward))
                self.place_obj(self.obstacles[i_obst], max_tries=100)

        # add transient obstacles
        self.penalties = []
        if self.transient_obstacles is not None:
            for location in self.transient_obstacles:
                self.penalties.append(
                    ObstacleNoTerminate(reward=self.transient_penalty)
                )
                self.grid.set(location[0], location[1], self.penalties[0])
        else:
            for i_obst in range(self.n_transient_obstacles):
                self.penalties.append(
                    ObstacleNoTerminate(reward=self.transient_penalty)
                )
                self.place_obj(self.penalties[i_obst], max_tries=100)

        # Place a goal square in the bottom-right corner
        if hasGoal is False:
            self.put_obj(Goal(), width - 2, height - 2)

        # # Place the agent
        # if self.agent_start_pos is not None:
        #     self.agent_pos = self.agent_start_pos
        #     self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return self.termination_reward
