import time
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils import aec_to_parallel
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import ProbabilisticModel
from line_profiler import LineProfiler

N = 25
# Start with AEC environment
aec_env = simple_spread_v3.env(N=N, max_cycles=10)
agent_ids = aec_env.possible_agents
model = ProbabilisticModel(agent_ids=agent_ids, failure_prob=0.3)
wrapped_env = CommunicationWrapper(aec_env, failure_models=[model])

# Convert to parallel for actual use
par_env = aec_to_parallel(wrapped_env)
par_env.reset(seed=42)

# 🟡 LINE PROFILER
lp = LineProfiler()

# Profile the AEC wrapper's observe method (this is what's used internally)
lp.add_function(wrapped_env.observe)

# Add internal masking logic from CommunicationWrapper
if hasattr(wrapped_env, "_apply_comm_mask"):
    lp.add_function(wrapped_env._apply_comm_mask)

# Add SharedObsWrapper.observe if present
shared_obs_wrapper = wrapped_env.env  # The wrapped environment is the SharedObsWrapper
if hasattr(shared_obs_wrapper, "observe"):
    lp.add_function(shared_obs_wrapper.observe)


# Profile parallel env step & reset (which use the observe methods internally)
@lp
def profile_parallel_env(env, steps=10):
    # Reset the environment
    observations, _ = env.reset(seed=42)

    # Take multiple steps
    for _ in range(steps):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, _, _, _, _ = env.step(actions)


# Alternative: Profile the AEC's observe directly
@lp
def profile_aec_observe(env, steps=10):
    env.reset(seed=42)
    for _ in range(steps):
        for agent in env.agents:
            _ = env.observe(agent)
        # Move to next agent
        action = env.action_space(env.agent_selection).sample()
        env.step(action)


# Run both profiling functions
print("Profiling parallel environment:")
profile_parallel_env(par_env)

print("\nProfiler for AEC environment observe methods:")
profile_aec_observe(wrapped_env)

lp.print_stats()