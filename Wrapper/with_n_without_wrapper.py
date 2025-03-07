import numpy as np
import pandas as pd
from tabulate import tabulate
from pettingzoo.mpe import simple_spread_v3
from Wrapper.comm_failure_api import CommunicationFailure

env = simple_spread_v3.parallel_env(max_cycles=25)
wrapped_env = CommunicationFailure(
    env,
    failure_prob=0.3,
    seed=42
)

obs, _ =env.reset(seed=42)
wrapped_obs, _ = wrapped_env.reset(seed=42)

print("Original Observations:")
for agent, o in obs.items():
    print(f"{agent}: {o}")

print("\nWrapped Observation (with potential failures): ")
for agent, o in wrapped_obs.items():
    print(f"{agent}: {o}")

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs, _, _, _, _ = env.step(actions)
wrapped_obs, _, _, _, _ = wrapped_env.step(actions)

print("\nOriginal Observations (after one step): ")
for agent, o in obs.items():
    print(f"{agent}: {o}")

print("\nWrapped Observation (after one step with potential failures): ")
for agent, o in wrapped_obs.items():
    print(f"{agent}: {o}")

env.close()
wrapped_env.close()

df_raw = pd.DataFrame.from_dict(obs, orient='index')
print(f"Obs without Wrapper {df_raw}")
df_norm = pd.DataFrame.from_dict(wrapped_obs, orient='index')
print(df_norm)