# ============ Markov and Signal Model ============ #
# Markov is used for temporally corelated communication failures
# Signal-based attenuation is added absed on distance and physics
# Realistic communication patterns are created


from mpe2 import simple_tag_v3
from failure_api.wrappers import CommunicationWrapper
from failure_api.communication_models import (BaseMarkovModel, SignalBasedModel)
import numpy as np
from pettingzoo.utils import aec_to_parallel
from utils.position_utils import make_position_fn





# Define Markov transition probabilities
# This creates temporal correlation in communication failures
markov_transitions = np.array([
    [0.7, 0.3],  # 70% chance to stay disconnected, 30% chance to become connected
    [0.1, 0.9]  # 10% chance to become disconnected, 90% chance to stay connected
])

# ---- Create environment
ms_env = simple_tag_v3.env(num_good=3, num_adversaries=2, num_obstacles=0)
agent_ids = ms_env.possible_agents

# ---- Apply Communication wrapper
wrapped_ms_env = CommunicationWrapper(
    ms_env,
    failure_models=[
        BaseMarkovModel(
            agent_ids=agent_ids,
            default_matrix=markov_transitions,
        ),
        SignalBasedModel(
            agent_ids=agent_ids,
            tx_power=150.0,
            min_strength=0.3,
            dropout_alpha=0.05,
            pos_fn=make_position_fn(ms_env, return_batch=True)
        )
    ]
)

# ---- Convert to parallel environment
par_wrapped_ms_env = aec_to_parallel(wrapped_ms_env)
observations, infos = par_wrapped_ms_env.reset(seed=42)

history = []

# ---- Run the Environment
for step in range(10):
    # ---- Sample random actions
    actions = {agent: par_wrapped_ms_env.action_space(agent).sample() for agent in par_wrapped_ms_env.agents}
    observations, rewards, terminations, truncations, infos = par_wrapped_ms_env.step(actions)

    # ---- Record communication state
    ms_comm_state = wrapped_ms_env.get_communication_state().astype(int)
    history.append(ms_comm_state.copy())

    # ---- Print statistics for this step
    active_connections = np.sum(ms_comm_state) - len(agent_ids)
    total_possible = len(agent_ids) * (len(agent_ids) - 1)

    print(
        f"Step {step}: {active_connections}/{total_possible} links active ({active_connections / total_possible:.1%})")

    # ---- Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        break

# ---- Analyze temporal patterns for a specific connection
if len(agent_ids) >= 2:
    agent_1, agent_2 = agent_ids[0], agent_ids[1]
    idx1, idx2 = 0, 1

    print(f"\nCommunication link {agent_1} → {agent_2} over time:")
    for step, state in enumerate(history):
        status = "Connected" if state[idx1, idx2] else "Disconnected"
        print(f"Step {step}: {status}")

    # Calculate how often the state changes (lower = stronger temporal correlation)
    changes = sum(1 for i in range(1, len(history)) if history[i][idx1, idx2] != history[i - 1][idx1, idx2])
    print(f"\nState changes: {changes} out of {len(history) - 1} transitions")
    print(f"Persistence rate: {1 - changes / (len(history) - 1):.1%}")

print("Completed")