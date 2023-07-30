from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from blue_ai.envs.custom_world_objects import GoalNoTerminate
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Wall
import numpy as np
import imageio
import os


class TransientGoals(MiniGridEnv):

    def __init__(
        self,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        termination_reward=1,
        transient_reward=0.25,
        n_transient_goals=3,
        img_filename='env1.png',
        transient_locations=None,
        **kwargs
    ):

        self.im = imageio.imread(os.path.join(os.path.dirname(__file__), img_filename))
        width = len(self.im[0])
        height = len(self.im)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.termination_reward = termination_reward
        self.transient_reward = transient_reward
        self.n_transient_goals = n_transient_goals
        self.transient_locations = transient_locations

        mission_space = MissionSpace(mission_func=self._gen_mission)

        max_steps = 4 * width ** 2

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )

    def step(self, action):
        self.step_count += 1

        mandatory = 0
        optional = 0
        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
                # reward -= 0.001
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
                mandatory += 1
            if fwd_cell is not None and fwd_cell.type == "goalNoTerminate":
                reward = fwd_cell.reward
                optional += 1
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup() and fwd_cell.type == "keyReward":
                # if self.carrying is None:
                reward = self._reward()
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
            elif fwd_cell and fwd_cell.can_pickup():
                # if self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)


        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        im = self.im

        self.grid = Grid(width, height)
        obj_type = Wall

        hasGoal = False
        hasStart = False

        # Generate the surrounding walls
        for x in range(0, width):
            for y in range(0, height):
                if np.sum(im[y][x]) == 255:
                    self.grid.set(x, y, obj_type())
                # goal 101 - 254
                if np.sum(im[y][x]) > 255 and np.sum(im[y][x]) < 510:
                    self.put_obj(Goal(), x, y)
                    hasGoal = True
                # start 1 - 100
                # if np.sum(im[y][x]) > 511 and np.sum(im[y][x]) < 1020 and np.sum(im[y][x]) != 594:
                # if np.sum(im[y][x]) > 511 and np.sum(im[y][x]) < 1020:
                #     # print("set start pos")
                #     self.agent_start_pos = (x, y)
                #     hasStart = True
                # if np.sum(im[y][x]) == 594:
                #     self.put_obj(KeyReward(), x, y)

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

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        return self.termination_reward