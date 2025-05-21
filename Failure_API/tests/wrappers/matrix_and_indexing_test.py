# test_communication_matrix_alignment.py

from mpe2 import simple_spread_v3
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.communication_models import DistanceModel
import numpy as np

def test_matrix_alignment():
    N = 3
    env = simple_spread_v3.env(N=N, max_cycles=5)
    agent_ids = env.possible_agents

    model = DistanceModel(
        agent_ids=agent_ids,
        distance_threshold=10.0,
        pos_fn=lambda agent: np.array([0.0, 0.0]),  # Dummy position function
        max_bandwidth=10.0
    )
    wrapper = CommunicationWrapper(env, failure_models=[model])

    obs, infos = wrapper.reset(seed=42)
    print("📏 Matrix shape:", wrapper.comms_matrix.matrix.shape)
    print("👥 Agents:", wrapper.agents)
    print("📌 agent_id_to_index:", wrapper.comms_matrix.agent_id_to_index)

    # --- Step 1: Assert matrix size matches number of agents
    matrix = wrapper.comms_matrix.matrix
    expected_shape = (N, N)

    assert matrix.shape == expected_shape, (
        f"[FAIL] Matrix shape {matrix.shape} does not match expected {expected_shape}"
    )

    # --- Step 2: Assert every agent in env.agents is in agent_to_index
    missing_agents = [a for a in wrapper.agents if a not in wrapper.comms_matrix.agent_id_to_index]
    assert not missing_agents, (
        f"[FAIL] These agents are missing in agent_id_to_index: {missing_agents}"
    )

    # --- Step 3: Assert the matrix is indexed correctly
    for sender in wrapper.agents:
        for receiver in wrapper.agents:
            try:
                i = wrapper.comms_matrix.agent_id_to_index[sender]
                j = wrapper.comms_matrix.agent_id_to_index[receiver]
                _ = matrix[i, j]  # access test
            except IndexError:
                raise AssertionError(f"[FAIL] Index out of bounds for pair ({sender}, {receiver})")

    print("✅ Communication matrix test passed. Matrix shape and indexing are correct.")

if __name__ == "__main__":
    test_matrix_alignment()
