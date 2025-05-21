##------- Basic Usage -------#
# SharedOBS wrapper is already instantiated in Communication wrapper
import numpy as np
import pandas as pd
from mpe2 import simple_spread_v3
from failure_api.wrappers import (CommunicationWrapper,
                                  NoiseWrapper)
from failure_api.noise_models import (GaussianNoiseModel)
from failure_api.communication_models import ProbabilisticModel, ActiveCommunication
from pettingzoo.utils import aec_to_parallel


# Step 1: Create base environment
env = simple_spread_v3.env(N=3, max_cycles=25)
agent_ids = env.possible_agents

# Step 2: Apply communication wrapper with probabilistic failures
# Each communication link has a 30% chance of failing
model = ProbabilisticModel(agent_ids, failure_prob=0.3)
wrapped_comm_env = CommunicationWrapper(env, failure_models=[model])

# Step 3: Apply noise to successfully transmitted observations
wrapped_ns_env = NoiseWrapper(wrapped_comm_env, GaussianNoiseModel(std=0.5))

# Step 4: Convert wrapper from aec mode to parallel mode
cv_wrapped = aec_to_parallel(wrapped_ns_env)

# Step 5: Run the environment 10 times
observations = cv_wrapped.reset(seed=42)
for _ in range(10):
    # Sample random actions
    actions = {agent: cv_wrapped.action_space(agent).sample() for agent in cv_wrapped.agents}

    # Step the environment
    observations, rewards, terminations, truncations, infos = cv_wrapped.step(actions)

    # Check agents that can communicate with each other
    comms = wrapped_ns_env.get_communication_state().astype(int)
    print(f"Step{_}, Number of active communication links: {np.sum(comms)}")

    # Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        break

print("Successful")