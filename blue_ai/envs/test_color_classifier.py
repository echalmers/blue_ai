import pytest
from blue_ai.envs.color_classifier import *


test_cases = {
    "test_black_color": ((0, 0, 0), Colors.BLACK),
    "test_white_color": ((255, 255, 255), Colors.WHITE),
    "test_red_color": ((255, 0, 0), Colors.RED),
    "test_green_color": ((0, 255, 0), Colors.GREEN),
    "test_blue_color": ((0, 0, 255), Colors.BLUE),
    "test_yellow_color": ((255, 255, 0), Colors.YELLOW),
    "test_cyan_color": ((0, 255, 255), Colors.CYAN),
    "test_magenta_color": ((255, 0, 255), Colors.MAGENTA),
}


@pytest.mark.parametrize(
    "input_rgb,expected", test_cases.values(), ids=test_cases.keys()
)
def test_color_classification(input_rgb, expected):
    result = classify_color(input_rgb)
    assert result == expected


def test_unknown_color():
    input_rgb = (128, 128, 128)
    with pytest.raises(
        ValueError,
        match="Color distance exceeds threshold, classification is not possible.",
    ):
        classify_color(input_rgb)


if __name__ == "__main__":
    pytest.main()
