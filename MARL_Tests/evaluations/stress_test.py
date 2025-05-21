import numpy as np
import pytest
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


@pytest.mark.parametrize("name,comm_matrix", [
    ("All connected", np.ones((3, 3), dtype=int)),
    ("Self only", np.eye(3, dtype=int)),
    ("Random partial", np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])),
    ("No communication", np.zeros((3, 3), dtype=int)),
])
def test_apply_comm_mask_stress(name, comm_matrix):
    agent_ids = ["agent_0", "agent_1", "agent_2"]
    dummy_obs = {
        "agent_0": np.full(5, 1.0),
        "agent_1": np.full(5, 2.0),
        "agent_2": np.full(5, 3.0),
    }

    class DummyEnv:
        def observe(self, agent): return dummy_obs.copy()
        def reset(self): pass
        @property
        def possible_agents(self): return agent_ids

    wrapper = CommunicationWrapper(env=DummyEnv(), failure_models=[lambda x: None])
    wrapper.agent_ids = agent_ids
    wrapper.comms_matrix.matrix = comm_matrix.copy()

    for receiver in agent_ids:
        obs_input = dummy_obs.copy()
        masked = wrapper._apply_comm_mask(obs_input.copy(), receiver)

        for sender in agent_ids:
            can_see = wrapper.comms_matrix.can_communicate(sender, receiver)
            expected = dummy_obs[sender] if (sender == receiver or can_see) else np.zeros_like(dummy_obs[sender])
            actual = masked[sender]

            assert np.allclose(actual, expected), (
                f"\n❌ [{name}] {receiver} received wrong masking for {sender}:\n"
                f"Expected: {expected}\nGot: {actual}\nComm Matrix:\n{comm_matrix}"
            )
