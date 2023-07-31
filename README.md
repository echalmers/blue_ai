# Custom minigrid environments

Authors: Jesse Viehweger, Emily Allerdings

this repository extends minigrid with the following environments:

- blue_ai_envs/TransientGoals
  - gridworld with a static large-reward goal, and several transient smaller-reward goals. The static goal appears in the same place in every episode, while the transient goals appear randomly.

### Installation and Demo
After cloning, install using `pip install .`

Demo the custom environment using `python manual_control.py --env blue_ai/TransientGoals`
