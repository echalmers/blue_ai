# Agent_Visualizer.py
# Jesse Viehweger, Eric Chalmers
# April 20, 2023

# Purpose:
# The purpose of this code is to be able to visualize how a trained agent
# is behaving within the Blue AI environment. 150 step will be taken in the environment
# to show the behaviour.

# Inputs:
# There are two inputs that this program can take.
# 1) The dropout associated with the agent you are wanting to visualize (default = 0).
# 2) Whether or not you want the visualization to be saved as a gif. For this a 0
#    does not save a gif, 1 does (default = 0). 
# Example Call: python3 Agent_Visualizer.py --drop 20 --gif 1

# Results:
# A gif will be produced if desired and will be stored in the Visualizations directory with a name
# indicating its dropout level.

# Assumptions:
# An agent must be trained and located in the Agents directorty with the dropout requested to be
# visualized in this program. 

import argparse
from pathlib import Path
import pickle
import gymnasium as gym
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

from pathlib import Path
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from Blue_AI_Trainer import image_to_vec
from minimal_DQN import MinimalDQN
from Blue_AI_Trainer import DQN
from Blue_AI_Trainer import TransitionMemory
from blue_ai_envs.envs.transient_goals import TransientGoals
import os

STEPS = 150
#gif fps
FPS = 5

#get desired dropout
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--drop", help="amount of dropout to be used in the model", default=0
)
parser.add_argument(
    "--gif", help="indicate if a gif of the agent should be recorded (0 = no, 1 = yes)", default=0
)
args = parser.parse_args()

drop = args.drop
drop = int(drop)

gif = args.gif
gif = bool(gif)


# Load the environment
env = gym.make('blue_ai_envs/TransientGoals', tile_size=32, render_mode='human')
env.reset(seed=None)

print("Environment loaded\n")

# Load agent
file1 = "agent" + str(drop) + ".pkl"
file1 = os.path.join('Agents', file1)
path1 = Path(file1)

if(path1.is_file()):
    infile = open(file1,'rb')
    agent = pickle.load(infile)
    infile.close()
    print("Reusing Trained Network")

print("Agent loaded\n")

# Run the agent

if gif:
    from array2gif import write_gif
    frames = []

state = env.reset()
state = image_to_vec(state[0]['image'])

for step in range(STEPS):
        # get & execute action chosen by DQN agent
        env.render()
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)
        new_state = image_to_vec(new_state['image'])

        # Saves frame for gif if desired
        if gif:
            frames.append(np.moveaxis(env.get_frame(), 2, 0))
        
        if done:
            
            state = env.reset()
            state = image_to_vec(state[0]['image'])
        else:
            state = new_state

# Saves visualized steps as a gif.
if gif:
    print("Saving gif... ", end="")

    if not os.path.exists('Visualizations'):
        os.makedirs('Visualizations')
    
    new_file_path = os.path.join('Visualizations', 'visualization' + str(drop) +'.gif')
    write_gif(np.array(frames), new_file_path, fps=FPS)
    
    print("Done.")
