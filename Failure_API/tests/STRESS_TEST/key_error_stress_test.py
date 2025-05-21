from pettingzoo.utils.conversions import aec_to_parallel
import unittest
import numpy as np
from mpe2 import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
from Failure_API.src.failure_api.wrappers.communication_wrapper import CommunicationWrapper
from Failure_API.src.failure_api.wrappers import SharedObsWrapper, BaseWrapper
from Failure_API.src.failure_api.wrappers.noise_wrapper import NoiseWrapper
from Failure_API.src.failure_api.communication_models import (ProbabilisticModel,
                                                              ActiveCommunication, BaseMarkovModel, DistanceModel)
from Failure_API.src.failure_api.noise_models.gaussian_noise import GaussianNoiseModel

#
# class TestCommunicationWrapperSync(unittest.TestCase):
#     def setUp(self):
#         self.agent_count = 3
#         self.env = simple_spread_v3.env(N=self.agent_count, max_cycles=5)
#         self.agent_ids = self.env.possible_agents.copy()
#
#         # Dummy position function — all agents at origin
#         def dummy_pos_fn(agent):
#             return np.array([0.0, 0.0])
#
#         self.model = DistanceModel(
#             agent_ids=self.agent_ids,
#             distance_threshold=1.0,
#             pos_fn=dummy_pos_fn,
#             failure_prob=0.0,
#             max_bandwidth=1.0
#         )
#
#         self.env = CommunicationWrapper(self.env, failure_models=[self.model])
#         self.env.reset(seed=42)
#
#     def test_no_keyerror_on_step_or_reset(self):
#         """Ensure communication wrapper maintains agent sync through steps."""
#         try:
#             for _ in range(3):
#                 agent = self.env.agent_selection
#                 act = self.env.action_space(agent).sample()
#                 self.env.step(act)
#         except KeyError as e:
#             self.fail(f"KeyError raised during env.step(): {e}")
#
#     def test_comms_matrix_reset_agent_id_mismatch(self):
#         """Check comms_matrix update fails gracefully on agent mismatch."""
#
#         # BAD model — intentionally wrong agent_ids
#         bad_model = DistanceModel(
#             agent_ids=["fake_agent_1", "fake_agent_2"],
#             distance_threshold=1.0,
#             pos_fn=lambda agent: np.array([0.0, 0.0])
#         )
#
#         # Standalone communication matrix with real agents
#         from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
#         comms_matrix = ActiveCommunication(agent_ids=["agent_0", "agent_1", "agent_2"])
#
#         # Direct call — do NOT pass through CommunicationWrapper
#         print(f"[DEBUG] Checking current_agents: {comms_matrix.agent_ids} vs model.agent_ids: {bad_model.agent_ids}")
#         with self.assertRaises(AssertionError):
#             bad_model.update_connectivity(comms_matrix)
#
#     def test_partial_dead_agents_during_update(self):
#         """Simulate agent drop mid-run and check robustness of update."""
#         try:
#             for step in range(5):
#                 agent = self.env.agent_selection
#                 act = self.env.action_space(agent).sample()
#                 self.env.step(act)
#
#                 # Artificially remove an agent halfway
#                 if step == 2 and len(self.env.agents) > 1:
#                     removed_agent = self.env.agents[-1]
#                     self.env.agents.remove(removed_agent)
#                     print(f"[DEBUG] Removed agent: {removed_agent}")
#
#                     # Update communication state manually
#                     self.env._update_communication_state()
#         except KeyError as e:
#             self.fail(f"KeyError due to stale agent list after removal: {e}")
#
#     def test_reset_alignment(self):
#         """Ensure agent_ids are synced properly after env reset."""
#         self.env.reset(seed=123)
#         try:
#             self.env._update_communication_state()
#         except KeyError as e:
#             self.fail(f"KeyError during comms_matrix reset after full env reset: {e}")
#

class TestModelMismatchStandalone(unittest.TestCase):
    def test_comms_matrix_reset_agent_id_mismatch(self):
        """Check comms_matrix update fails gracefully on agent mismatch."""

        from Failure_API.src.failure_api.communication_models.distance_based_model import DistanceModel
        from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

        # BAD model — intentionally wrong agent_ids
        bad_model = DistanceModel(
            agent_ids=["fake_agent_1", "fake_agent_2"],
            distance_threshold=1.0,
            pos_fn=lambda agent: np.array([0.0, 0.0])
        )

        # Communication matrix with actual agents
        comms_matrix = ActiveCommunication(agent_ids=["agent_0", "agent_1", "agent_2"])

        # Confirm mismatch
        print(f"[DEBUG] Checking current_agents: {comms_matrix.agent_ids} vs model.agent_ids: {bad_model.agent_ids}")

        with self.assertRaises(AssertionError):
            bad_model.update_connectivity(comms_matrix)


if __name__ == "__main__":
    unittest.main()
