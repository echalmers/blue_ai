from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (fill_coords, point_in_rect)
from minigrid.core.constants import COLORS, OBJECT_TO_IDX


class GoalNoTerminate(WorldObj):
    def __init__(self, reward=0.5):
        super().__init__("goalNoTerminate", "blue")
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


OBJECT_TO_IDX['goalNoTerminate'] = len(OBJECT_TO_IDX)