# Building an Artificial Intelligence Model of Post Traumatic Stress Disorder

A python implementation of a Reinforcement Learning approach to model PTSD. This work will be the base fot the paper "Building an Artificial Intelligence Model of Post Traumatic Stress Disorder" (provisional title)

## Table of Contents

* [Project Goal](#project-goal)
* [Implemented Core Components](#implemented-core-components)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Future Work](#future-work)
* [Project Members](#project-members)

## Project Goal

This project was conducted during a research internship organized by Mitacs at the Mount Royal University, and extends the prior work from the same lab. The primary goal was to gain a deeper understanding of PTSD in the brain. 

The initial objective was to expose multiple Machine Learning agents with different cognitive impairments to an environment where they are challenged to find the path to a goal. Different simulated cognitive impairments lead to different bahaviours of the agent, where we want to draw conclusion about PTSD.

## Implemented Core Components

The following components have been implemented:

- **Agents:**
    - HealtyAgent: Our agents which simulates a healty brain structure
    - PTSDAgent: Increased weight decay of 0.003 simulating assumed spine loss in PTSD patients
    - TraumaSynapticDeficitAgent: The increased weight decay is active after the trauma got induced to the agent
    
    -> All of these agents are based on a DQN architecture.

- **Environments:**
    - TransientGoals: Inherits from [MiniGridEnv](https://minigrid.farama.org/)
    - Image2VecWrapper: Wrapper to create the input for the DQN network

- **Scripts:**
    - execute: Main script which starts a full trial and calls all the following functions/scripts
    - train_agents: Train the agents for a given number of steps in a given environment
    - view_performance: Plot the cumulative rewards obtained by the agents during a trial and the objects reached per episode
    - induce_traumatic_event: Exposes the agents to an traumatic event, where they are forced to perform an action leading to a high negative reward
    - investigate_Qvalues: Plot the Qvalues of the agents in a given state, which represents the willingness of the agents to perform the action (turn left, turn right, move forward, hide/do nothing)
    - train_interpretation_models: Train an additional model which reconstructs the state the agents see, just by getting the activations as input
    - view_reconstruction_loss: Plot the loss during the training of the interpretation models
    - view_reconstruction: Interactively move through an environment and plot the reconstructions, and plot a statistical analysis of the reconstructions
    - post_ptsd: Re-exposes the agents to the situation of the trauma, but this time without the negative reward (exposure therapy)
    - test_performance: Test the current performance of the agents

## Project Structure:

```
blue_ai/
│
├── blue_ai/                     # Source code for the modeling of various cognitive impairments
│   ├── __init__.py       
│   ├── agents/                  # Subfolder for the agents
│   │   ├── __init__.py
│   │   ├── abstract.py          # The abstract agent
│   │   ├── agent_classes.py     # Implementation of the BaseAgent and all other specifications
│   │   └── dqn.py               # The core DQN implementation for the agents
│   │
│   ├── envs/                    # Subfolder for the environments
│   │   ├── __init__.py
│   │   ├── color_classifier.py  
│   │   ├── custom_decay.py  
│   │   ├── custom_world_objects.py  
│   │   ├── custom_wrappers.py   # Wrapper for the envs which creates the input for the DQN
│   │   ├── env1.png             # picture which can be used for environment creation
│   │   ├── test_color_classifier.py
│   │   └── transient_goals.py   # Creating the envs and handling the actions
│   │
│   ├── scripts/                 # Subfolder for the scripts
│   │   ├── __init__.py
│   │   ├── data/                # Folder collecting all important agent pkl files and plots
│   │   │   └── example_trial/   # All the data from one full trial
│   │   │       └── img/         # Plots and screenshots from a trial
│   │   │       
│   │   ├── img/                 # Image folder (which is NOT IN USE for the PTSD project)
│   │   │       
│   │   ├── ptsd/                # Scripts for the modeling of PTSD
│   │   │   ├── experiments/     
│   │   │   │   └── execute.py   # Main script which runs a full trial and saves all results
│   │   │   │    
│   │   │   ├── __init__.py  
│   │   │   ├── induce_traumatic_event.py 
│   │   │   ├── investigate_Qvalues.py 
│   │   │   ├── post_ptsd.py 
│   │   │   ├── README.md        # PTSD project documentation (you are reading this) 
│   │   │   ├── test_performance.py 
│   │   │   ├── train_agents.py 
│   │   │   ├── train_interpretation_models.py 
│   │   │   ├── view_forward_Qvalues.py
│   │   │   ├── view_performance.py  
│   │   │   ├── view_reconstruction_loss.py 
│   │   │   └── view_reconstruction.py
│   │   │   
│   │   ├── schrizophrenia/      # Scripts for the modeling of Schizophrenia
.   .   .                        
.   .   .                        # Scripts for the modeling of Depression
.   .   .
│   └── utils.py        
│
├── .gitignore                
├── environment_demo          # Interactive demo of the used environment
├── README.md                 # Project documentation (you are reading the README inside the ptsd folder)
└── setup.py                  # Project setup
```


## Usage
1. Navigate to the blue_ai folder in your terminal
2. If you wish to run a whole trial with all the implemented functionalities, go to the execute.py script in the experiments folder and type the following command in your terminal:
    ```bash
        python -m blue_ai.scripts.ptsd.experiments.execute <directory>
    ```
    For 'directory' insert the name of a folder, which will be created to save the results and plots.
3. It is also possible to run the single scripts independently. Simply type:
    ```bash
        python -m blue_ai.scripts.ptsd.<script_name>
    ```
    followed by the necessary system arguments.


## Future Work

In future implementations we want to introduce a three model approach containing representation of the hippocampus, prefrontal cortex and amygdala.

## Project Members

- [@echalmers](https://github.com/echalmers)
- [@benitomano](https://github.com/benitomano)