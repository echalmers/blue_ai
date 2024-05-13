from typing import Tuple
import numpy as np


class ColorClassifier:
    COLORS = {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
    }
    THRESHOLD = 100

    @staticmethod
    def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        return np.linalg.norm(np.array(c1) - np.array(c2))

    @staticmethod
    def classify_color(rgb: Tuple[int, int, int]) -> str:
        distances = {
            color: ColorClassifier.color_distance(rgb, color_rgb)
            for color, color_rgb in ColorClassifier.COLORS.items()
        }

        closest_color = min(distances, key=distances.get)

        if distances[closest_color] <= ColorClassifier.THRESHOLD:
            return closest_color

        raise ValueError(
            "Color distance exceeds threshold, classification is not possible."
        )
