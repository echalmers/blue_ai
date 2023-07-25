# Tasks_Completed_Visualizer.py
# Jesse Viehweger, Eric Chalmers
# April 20, 2023

# Purpose:
# This program visualizes how the tasks chosen by an agent change as dropout increase. It 
# displays the tasks complete by an agent with 0%, 25%, 50%, 75%, and 100% dropout. The tasks show
# are either optional tasks (in orange), or mandatory tasks (in blue).

# Input:
# The program takes the logs from the agents with 0%, 25%, 50%, 75%, and 100% dropout for the Logs directory
# and shows the coresponding bargraphs for the data.


import os
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def getChange(new, old):
    result = (new - old)/old
    return result

file5 = "Log" + str(100) + ".pkl"
file5 = os.path.join('Logs', file5)
path5 = Path(file5)
if(path5.is_file()):
    infile = open(file5,'rb')
    logs5 = pickle.load(infile)
    infile.close()
    print("Adding to saved log")

file4 = "Log" + str(75) + ".pkl"
file4 = os.path.join('Logs', file4)
path4 = Path(file4)
if(path4.is_file()):
    infile = open(file4,'rb')
    logs4 = pickle.load(infile)
    infile.close()
    print("Adding to saved log")

file3 = "Log" + str(50) + ".pkl"
file3 = os.path.join('Logs', file3)
path3 = Path(file3)
if(path3.is_file()):
    infile = open(file3,'rb')
    logs3 = pickle.load(infile)
    infile.close()
    print("Adding to saved log")

file2 = "Log" + str(25) + ".pkl"
file2 = os.path.join('Logs', file2)
path2 = Path(file2)
if(path2.is_file()):
    infile = open(file2,'rb')
    logs2 = pickle.load(infile)
    infile.close()
    print("Adding to saved log")


file1 = "Log" + str(0) + ".pkl"
file1 = os.path.join('Logs', file1)
path1 = Path(file1)
if(path1.is_file()):
    infile = open(file1,'rb')
    logs1 = pickle.load(infile)
    infile.close()
    print("Adding to saved log")

mandatory = []
optional = []

df1 = pd.DataFrame.from_dict(logs1["mandatory_rewards"])
df1a = pd.DataFrame.from_dict(logs1["optional_rewards"])

df1 = df1.iloc[-1]
df1 = df1.iloc[0]
df1a = df1a.iloc[-1]
df1a = df1a.iloc[0]


print(df1)

mandatory.append(df1)
optional.append(df1a)

df2 = pd.DataFrame.from_dict(logs2["mandatory_rewards"])
df2a = pd.DataFrame.from_dict(logs2["optional_rewards"])

df2 = df2.iloc[-1]
df2 = df2.iloc[0]
df2a = df2a.iloc[-1]
df2a = df2a.iloc[0]

mandatory.append(df2)
optional.append(df2a)

df3 = pd.DataFrame.from_dict(logs3["mandatory_rewards"])
df3a = pd.DataFrame.from_dict(logs3["optional_rewards"])

df3 = df3.iloc[-1]
df3 = df3.iloc[0]
df3a = df3a.iloc[-1]
df3a = df3a.iloc[0]

mandatory.append(df3)
optional.append(df3a)

df4 = pd.DataFrame.from_dict(logs4["mandatory_rewards"])
df4a = pd.DataFrame.from_dict(logs4["optional_rewards"])

df4 = df4.iloc[-1]
df4 = df4.iloc[0]
df4a = df4a.iloc[-1]
df4a = df4a.iloc[0]

mandatory.append(df4)
optional.append(df4a)

df5 = pd.DataFrame.from_dict(logs5["mandatory_rewards"])
df5a = pd.DataFrame.from_dict(logs5["optional_rewards"])

df5 = df5.iloc[-1]
df5 = df5.iloc[0]
df5a = df5a.iloc[-1]
df5a = df5a.iloc[0]

mandatory.append(df5)
optional.append(df5a)

i = 1 
changeMandatory = []
changeOptional = []
while (i < len(mandatory)):
    current = getChange(mandatory[i], mandatory[0])
    current = round(current,3)
    print("mandatory " + str(current))
    changeMandatory.append(current)
    current = getChange(optional[i], optional[0])
    current = round(current,3)
    print("optional " + str(current))
    changeOptional.append(current) 
    i = i + 1

def valuelabel(labels,changeMandatory):
    for i in range(len(labels) - 1):
        plt.text(i,changeMandatory[i],changeMandatory[i], ha = 'center',
                 bbox = dict(facecolor = 'cyan', alpha =0.8))

width = 0.3
plt.bar(np.arange(len(mandatory)), mandatory, width=width, label="Mandatory Tasks")
plt.bar(np.arange(len(optional))+ width, optional,  width=width, label="Optional Tasks")
labels = ['0%', '25%', '50%', '75%', '100%']
plt.xticks(range(len(mandatory)), labels)
plt.legend(loc='upper center')
plt.xlabel("Level of Depression")
plt.ylabel("Number of Tasks Completed")
plt.show()
