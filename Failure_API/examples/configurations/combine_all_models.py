from Failure_API.src.failure_api.wrappers import CommunicationWrapper, NoiseWrapper
from Failure_API.src.failure_api.communication_models import (
    DistanceModel, BaseMarkovModel, DelayBasedModel, SignalBasedModel
)
from Failure_API.src.failure_api.noise_models import GaussianNoiseModel, LaplacianNoiseModel
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils import aec_to_parallel

#---- Helper function

def get_positions(com_env):
    positions = {}
    for agent_id in com_env.agents:
        obs = com_env.observe(agent_id)
        if isinstance(obs, np.ndarray) and len(obs) >= 2:
            positions[agent_id] = obs[:2]
    return positions

#------- Create environment
com_env = simple_spread_v3.env(N=5, local_ratio=0.5)
agent_ids = com_env.possible_agents

#-------  Wrap environment
wrapped_com_env = CommunicationWrapper(
    com_env,
    failure_models=[
        # 1. Distance-based model
        DistanceModel(
            agent_ids=agent_ids,
            distance_threshold=1.0,
            pos_fn=lambda: get_positions(com_env),
            failure_prob=0.05
        ),
        # 2. Signal-based model
        SignalBasedModel(
            agent_ids=agent_ids,
            pos_fn=lambda: get_positions(com_env),
            tx_power=15.0,
            min_strength=0.05,
            dropout_alpha=0.2,
        ),
        BaseMarkovModel(
            agent_ids=agent_ids,
            default_matrix=np.array([
                [0.8, 0.2],  # 80% chance to stay disconnected, 20% to recover
                [0.05, 0.95] # 95% chance to stay connected, 5% chance to fail
            ])
        ),
        # 3. Delay-based model
        DelayBasedModel(
            agent_ids=agent_ids,
            min_delay=1,
            max_delay=3,
            message_drop_probability=0.05
        )
    ]
)

#------- Noise wrapper
ns_wrapped_com_env = NoiseWrapper(
    wrapped_com_env,
    noise_model=GaussianNoiseModel(std=0.02)
)

#-------- Convert to parallel environment
par_ns_wrapped_com_env = aec_to_parallel(ns_wrapped_com_env)
observations, infos = par_ns_wrapped_com_env.reset()
print(f"Environment has {len(agent_ids)} agents")

#-------- Run environment
for step in range(20):
    # -- Sample random actions
    actions = {agent: par_ns_wrapped_com_env.action_space(agent).sample() for agent in par_ns_wrapped_com_env.agents}
    observations, rewards, terminations, truncations, infos = par_ns_wrapped_com_env.step(actions)

    # -- Get communication state
    comm_state = ns_wrapped_com_env.get_communication_state().astype(int)
    active_links = np.sum(comm_state)
    max_links = len(agent_ids) * (len(agent_ids) - 1)

    print(f"Step {step}: {active_links}/{max_links} communication links active ({active_links / max_links:.1%})")

    # Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        break

print("Simulation completed")
