import pytest
from blue_ai.envs.color_classifier import ColorClassifier


def test_black_color():
    input_rgb = (0, 0, 0)
    expected = "Black"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_white_color():
    input_rgb = (255, 255, 255)
    expected = "White"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_red_color():
    input_rgb = (255, 0, 0)
    expected = "Red"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_green_color():
    input_rgb = (0, 255, 0)
    expected = "Green"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_blue_color():
    input_rgb = (0, 0, 255)
    expected = "Blue"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_yellow_color():
    input_rgb = (255, 255, 0)
    expected = "Yellow"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_cyan_color():
    input_rgb = (0, 255, 255)
    expected = "Cyan"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_magenta_color():
    input_rgb = (255, 0, 255)
    expected = "Magenta"
    result = ColorClassifier.classify_color(input_rgb)
    assert result == expected


def test_unknown_color():
    input_rgb = (128, 128, 128)
    with pytest.raises(
        ValueError,
        match="Color distance exceeds threshold, classification is not possible.",
    ):
        ColorClassifier.classify_color(input_rgb)


if __name__ == "__main__":
    pytest.main()
