"""
Demonstrates how observations transform through the wrapper chain.

This example shows:
1. Original observation from the base environment
2. Changes after communication filtering
3. Changes after noise application
4. Final structured observation format
"""

import numpy as np
from pettingzoo.butterfly import pistonball_v6
from Failure_API.src.failure_api.wrappers import (
    CommunicationWrapper, NoiseWrapper, SharedObsWrapper
)
from Failure_API.src.failure_api.communication_models import ProbabilisticModel
from Failure_API.src.failure_api.noise_models import GaussianNoiseModel

# Create base environment
env = pistonball_v6.parallel_env(n_pistons=3)
agent_ids = env.possible_agents

# Save unwrapped env for comparison
unwrapped_env = env

# Apply wrappers
env = CommunicationWrapper(
    env,
    failure_models=[ProbabilisticModel(agent_ids, failure_prob=0.5)]
)
env = NoiseWrapper(env, GaussianNoiseModel(std=0.1))
env = SharedObsWrapper(env)

# Reset environments
unwrapped_env.reset(seed=42)
observations = env.reset(seed=42)

# Trace observation flow for first agent
agent = agent_ids[0]
print(f"Tracing observation flow for {agent}")

# Step 1: Original observation
original_obs = unwrapped_env.observe(agent)
print(f"\n1. Original observation from base environment:")
print(f"   Shape: {original_obs.shape}")
print(f"   Mean value: {np.mean(original_obs):.4f}")

# Step 2: After communication filtering
# Force a communication failure for demonstration
env.comms_matrix.update(agent_ids[1], agent, 0.0)

# Get dictionary with agent_1's info zeroed out
comm_obs = env._apply_comm_mask(
    {a: unwrapped_env.observe(a) for a in agent_ids},
    agent
)
print(f"\n2. After communication filtering:")
print(f"   {agent_ids[0]} → {agent}: {'visible' if np.any(comm_obs[agent_ids[0]]) else 'masked'}")
print(f"   {agent_ids[1]} → {agent}: {'visible' if np.any(comm_obs[agent_ids[1]]) else 'masked'}")
print(f"   {agent_ids[2]} → {agent}: {'visible' if np.any(comm_obs[agent_ids[2]]) else 'masked'}")

# Step 3: After noise application
noise_model = GaussianNoiseModel(std=0.1)
noise_model.set_rng(np.random.RandomState(42))

noisy_obs = {}
for k, v in comm_obs.items():
    if np.any(v):  # Only apply noise to non-zero observations
        noisy_obs[k] = noise_model.apply(v)
    else:
        noisy_obs[k] = v

print(f"\n3. After noise application:")
mean_deviation = np.mean(np.abs(noisy_obs[agent] - comm_obs[agent]))
print(f"   Mean noise level: {mean_deviation:.4f}")

# Step 4: Final shared observation
final_obs = env.observe(agent)
print(f"\n4. Final shared observation:")
print(f"   Type: {type(final_obs)}")
print(f"   Keys: {list(final_obs.keys())}")
print(f"   Shapes: {[v.shape for v in final_obs.values()]}")