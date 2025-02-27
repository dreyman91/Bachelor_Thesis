# %%
import pettingzoo.mpe.simple_spread_v3 as simple_spread
import pandas as pd
from tabulate import tabulate
import numpy as np

# Create environment
env = simple_spread.env(render_mode="human")
env.reset()

# Run a few steps in the environment
for _ in range(500):  # Run for 100 steps
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None  # No action if agent is done
        else:
            action = env.action_space(agent).sample()  # Random action

        env.step(action)

env.close()

print(tabulate(env.observation_space, headers="keys", tablefmt="fancy_grid"))