# **Understanding the Market AEC Environment**

This custom environment MARL environment is built using **PettingZoo's AEC API**

## Imports
`from pettingzoo import AECEnv`

`from pettingzoo.utils import wrappers`

`import numpy as np`

`import gym`

`from gym import spaces`

`AECEnv` - The base class for Agent-Environment Cycle environments in PettingZoo. Since this is a turn-based MARL env, we must extend this class.

`wrappers`- To wrap the environment for monitoring and debugging