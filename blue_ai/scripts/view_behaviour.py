import polars as pl
import seaborn as sns
import numpy as np
import torch

from constants import DATA_PATH, FIGURE_PATH


def main():
    data = pl.read_parquet(DATA_PATH / "")
