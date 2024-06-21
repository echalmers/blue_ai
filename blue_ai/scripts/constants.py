from pathlib import Path

_CURRENT_DIR = Path(__file__).parent

N_TRIALS = 1

FIGURE_PATH = _CURRENT_DIR / "img"
DATA_PATH = _CURRENT_DIR / "data"


DATA_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
