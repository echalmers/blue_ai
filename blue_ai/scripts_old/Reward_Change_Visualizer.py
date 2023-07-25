# Reward_Change_Visualizer.py
# Jesse Viehweger, Eric Chalmers
# April 20, 2023

# Purpose:
# This programs purpose is to compare how the cumulative reward of an agent during training
# changes as the dropout level is increased.

# Input:
# The program takes the logs from agent training ranging from 0-100 dropout.

# Assumptions:
# It is assumed that all 101 logs are located in the Logs directory before running the program.

from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import os
import pickle

NUMLOGS = 101

# directory containing the pickle files
dir_path = "Logs"

# loop through all the pickle files in the directory
data = []
for i in range(0, NUMLOGS):
    file_path = os.path.join(dir_path, f"Log{i}.pkl")
    with open(file_path, "rb") as f:
        current = pickle.load(f)
        current = pd.DataFrame.from_dict(current["reward_per_episode"])
        current = current.cumsum()
        current = current.iloc[-1]
        data.append(current)

plt.plot(data)
plt.xlabel("Dropout")
plt.ylabel("Total Reward")


plt.show()