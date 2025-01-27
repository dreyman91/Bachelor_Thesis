import pettingzoo.mpe as mpe

import numpy as np
import pandas as pd

env = mpe.simple_tag_v3.env(render_mode=None)
env.reset()

agent = env.agent_selection

initial_observation = env.observe(agent)

action = env.action_space(agent).sample()
env.step(action)

new_observation = env.observe(agent)

def format_observation(observation):
    """Converts observation to an array of structured form"""
    return pd.DataFrame(observation.reshape(1, -1), columns=[f"Feature_{i}" for i in range(len(observation))])

print("Observation Analysis**")
print(f"Agent: {agent}\n")
print("Initial Observation")
print(format_observation(initial_observation))
print("New Observation (After action)")
print(format_observation(new_observation))
