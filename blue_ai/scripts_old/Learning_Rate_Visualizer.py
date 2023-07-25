# Learning_Rate_Visualizer.py
# Jesse Viehweger, Eric Chalmers
# April 20, 2023

# Purpose:
# This programs purpose is to compare the learning rate of 4 different agens.
# It compares the logs from a  0%, 25%, 50%, and 100% dropout agent and shows
# the learning curve of the agents.

# Input:
# It is required that the appropriate logs are stored in the Logs directory.


import os
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import pandas as pd

file4 = "Log" + str(100) + ".pkl"
file4 = os.path.join('Logs', file4)
path4 = Path(file4)
if(path4.is_file()):
    infile = open(file4,'rb')
    logs4 = pickle.load(infile)
    infile.close()

file3 = "Log" + str(50) + ".pkl"
file3 = os.path.join('Logs', file3)
path3 = Path(file3)
if(path3.is_file()):
    infile = open(file3,'rb')
    logs3 = pickle.load(infile)
    infile.close()
  
file2 = "Log" + str(25) + ".pkl"
file2 = os.path.join('Logs', file2)
path2 = Path(file2)
if(path2.is_file()):
    infile = open(file2,'rb')
    logs2 = pickle.load(infile)
    infile.close()
   
file1 = "Log" + str(0) + ".pkl"
file1 = os.path.join('Logs', file1)
path1 = Path(file1)
if(path1.is_file()):
    infile = open(file1,'rb')
    logs1 = pickle.load(infile)
    infile.close()
   
df1 = pd.DataFrame.from_dict(logs1["reward_per_episode"])
df2 = pd.DataFrame.from_dict(logs2["reward_per_episode"])
df3 = pd.DataFrame.from_dict(logs3["reward_per_episode"])
df4 = pd.DataFrame.from_dict(logs4["reward_per_episode"])

plt.plot(df1.cumsum(), label="Dropout = 0%")
plt.plot(df2.cumsum(), label="Dropout = 25%")
plt.plot(df3.cumsum(), label="Dropout = 50%")
plt.plot(df4.cumsum(), label="Dropout = 100%")

plt.legend(loc='upper center')

plt.xlabel("Episode")
plt.ylabel("Total Episode Reward")

plt.show()
