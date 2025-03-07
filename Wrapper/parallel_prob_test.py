
# PROBABILISTIC MODEL TEST

from pettingzoo.mpe import simple_spread_v3
from Wrapper.comm_failure_api import CommunicationFailure
from pettingzoo.utils import aec_to_parallel
env = simple_spread_v3.parallel_env()

wrapped_env = CommunicationFailure(
    env,
    comms_model="probabilistic",
    failure_prob=0.3,
    seed=42
)

observations, infos = wrapped_env.reset()
max_cycles = 25

for step in range(max_cycles):
    actions = {}
    for agent in wrapped_env.agents:
        actions = {agent: wrapped_env.action_space(agent).sample() for agent in wrapped_env.agents}

    observations, rewards, terminations, truncations, infos = wrapped_env.step(actions)
    if all(terminations.values()) or all(truncations.values()):
        break
wrapped_env.close()