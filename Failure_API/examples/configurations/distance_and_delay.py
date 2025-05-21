# ---------- Distance Model and Delay Based Model ----------#
# This example demonstrates how to wrap a PettingZoo environment with multiple communication failure models,
# showcasing how DistanceModel and DelayBasedModel can be
# composed to simulate realistic communication constraints among agents.

import numpy as np
from mpe2 import simple_tag_v3
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import (DelayBasedModel, DistanceModel)
from pettingzoo.utils import aec_to_parallel


# Helper function to extract agent positions from environment state
def get_positions(env):
    positions = {}
    for agent_id in env.agents:
        # Agent positions are the first two elements of the observation
        obs = env.observe(agent_id)
        if isinstance(obs, np.ndarray) and len(obs) >= 2:
            positions[agent_id] = obs[:2]  # x, y positions
        print(f"Agent Positions: {positions}")
    return positions


# Create base environment
env = simple_tag_v3.env(num_good=2, num_adversaries=2)
agent_ids = env.possible_agents

# Apply communication wrapper with multiple failure models
wrapped_env = CommunicationWrapper(
    env,
    failure_models=[
        # Model 1: Distance-based model - agents can only communicate within 0.5 units
        # Plus 10% random failure chance even when in range
        DistanceModel(
            agent_ids=agent_ids,
            distance_threshold=0.5,
            pos_fn=lambda: get_positions(env),
            failure_prob=0.1
        ),

        # Model 2: Delay-based model - messages take ake 1-3 timesteps to arrive
        # With 5% chance of complete message loss
        DelayBasedModel(
            agent_ids=agent_ids,
            min_delay=1,
            max_delay=3,
            message_drop_probability=0.05
        )
    ]
)

# Convert to parallel mode
par_wrapped_env = aec_to_parallel(wrapped_env)

# Run the environment
observations = par_wrapped_env.reset(seed=42)

for step in range(10):
    # Sample random actions
    actions = {agent: par_wrapped_env.action_space(agent).sample() for agent in par_wrapped_env.agents}

    # Step the environment
    observations, rewards, terminations, truncations, info = par_wrapped_env.step(actions)

    # Print communication state, This must be called on the wrapped env before conversion to parallel mode
    double_failure_comm = wrapped_env.get_communication_state().astype(int)

    # Check which agents are currently in the communication range

    print(f"\nStep {step}, Communication state:")
    for i, sender in enumerate(agent_ids):
        for j, receiver in enumerate(agent_ids):
            if i != j:
                status = "✅" if double_failure_comm[i, j] else "⛔"
                print(f" {sender} -> {receiver}, status: {status}")

    # Check if an episode is done
    if all(terminations.values()) or all(truncations.values()):
        break
print("\n========================""\nExample ran successfully")
