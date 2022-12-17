
MiniGridCustom

# Installation

To install the MinigridCustom library use `pip3 install -e .`.

# Demo Run

To demo run MinigridCustom run the following commands.
`cd minigrid`
`python3 manual_control.py --env MiniGrid-Custom-TestEnv`

# Changes Made to Original Minigrid

/minigrid/minigrid_env.py
- Reward range increased to accomadate optional rewards.
- Added goalNoTerminate object which acts as the optional reward with reduced different reward formula (smaller reward).

/minigrid/envs
- Added testenv which creats an environment based on a given png file. PNG files are located in /minigrid/ and the loaded env is named env1.png.


# original Minigrid repo:
Original repo [here](https://github.com/Farama-Foundation/Minigrid/issues/new/choose)
