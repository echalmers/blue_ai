from typing import Tuple
import numpy as np
from enum import Enum


class Colors(Enum):
    BLACK = 0
    WHITE = 1
    RED = 2
    GREEN = 3
    BLUE = 4
    YELLOW = 5
    CYAN = 6
    MAGENTA = 7


_COLORS = {
    (0, 0, 0): Colors.BLACK,
    (255, 255, 255): Colors.WHITE,
    (255, 0, 0): Colors.RED,
    (0, 255, 0): Colors.GREEN,
    (0, 0, 255): Colors.BLUE,
    (255, 255, 0): Colors.YELLOW,
    (0, 255, 255): Colors.CYAN,
    (255, 0, 255): Colors.MAGENTA,
}
COLOR_KEYS = np.array(list(_COLORS.keys()))
THRESHOLD = 150


def color_distance(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(c1 - c2, axis=1)


def classify_color(rgb: Tuple[int, int, int]) -> Colors:
    rgb_array = np.array(rgb)

    distances = color_distance(COLOR_KEYS, rgb_array)
    closest_index = np.argmin(distances)
    closest_color = _COLORS[tuple(COLOR_KEYS[closest_index])]
    certainty = distances[closest_index]

    if certainty <= THRESHOLD:
        return closest_color

    raise ValueError(
        "Color distance exceeds threshold, classification is not possible."
        f"Most likely is color {closest_color}, with certainty {certainty}"
    )
