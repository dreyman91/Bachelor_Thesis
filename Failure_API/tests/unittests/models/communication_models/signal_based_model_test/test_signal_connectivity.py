#!/usr/bin/env python
"""
Connectivity Tests for SignalBasedModel

This test file focuses on testing the connectivity determination logic
of the SignalBasedModel class. The model determines connectivity based on:
1. Signal strength threshold (min_strength parameter)
2. Probabilistic packet loss based on distance (dropout_alpha parameter)

We verify that:
1. Connections are properly established/broken based on signal strength
2. Probabilistic packet loss behaves correctly with distance
3. Transmit power affects connection range as expected
4. Self-connections are handled properly
"""

import unittest
from unittest.mock import Mock, MagicMock, call
import numpy as np

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


class TestConnectivityDetermination(unittest.TestCase):
    def setUp(self):
        self.agent_ids = ["agent1", "agent2", "agent3"]
        self.positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([10.0, 0.0, 10.0])
        }
        self.pos_fn = Mock(return_value=self.positions)
        self.model = SignalBasedModel(self.agent_ids, self.pos_fn)
        self.comms_matrix = Mock(spec=ActiveCommunication)
        self.comms_matrix.update = MagicMock()

    def test_signal_strength_threshold(self):
        mock_rng = Mock()
        mock_rng.random.return_value = 0.0
        self.model.rng = mock_rng

        self.model.update_connectivity(self.comms_matrix)
        update_calls = self.comms_matrix.update.call_args_list

        self.assertIn(call("agent1", "agent2", True), update_calls)
        self.assertIn(call("agent1", "agent3", True), update_calls)

    def test_transmit_power_effect(self):
        self.comms_matrix = Mock(spec=ActiveCommunication)
        self.comms_matrix.update = MagicMock()

        high_power_model = SignalBasedModel(self.agent_ids, self.pos_fn, tx_power=15.0)
        mock_rng = Mock()
        mock_rng.random.return_value = 0.0
        high_power_model.rng = mock_rng

        high_power_model.update_connectivity(self.comms_matrix)
        update_calls = self.comms_matrix.update.call_args_list
        print("Update Calls:", update_calls)

        self.assertIn(call("agent1", "agent3", True), update_calls)

    def test_min_strength_effect(self):
        high_threshold_model = SignalBasedModel(self.agent_ids, self.pos_fn, min_strength=0.5)
        mock_rng = Mock()
        mock_rng.random.return_value = 0.0
        high_threshold_model.rng = mock_rng

        high_threshold_model.update_connectivity(self.comms_matrix)
        update_calls = self.comms_matrix.update.call_args_list

        self.assertIn(call("agent1", "agent2", True), update_calls)
        self.assertIn(call("agent2", "agent3", False), update_calls)

    def test_probabilistic_packet_loss(self):

        n_connections = len(self.agent_ids) * (len(self.agent_ids) - 1)
        mock_values = [0.1, 0.9] * (n_connections // 2 + 1)

        mock_rng = Mock()
        mock_rng.random.side_effect = mock_values
        self.model.rng = mock_rng

        self.model.update_connectivity(self.comms_matrix)
        update_calls = self.comms_matrix.update.call_args_list

        # Check for at least one successful connection
        self.assertTrue(
            any(call_args[0][2] for call_args in update_calls),
            "At least one connection should succeed with low random values."
        )

        agent1_to_agent2_success = any(
            c == call("agent1", "agent2", True) for c in update_calls
        )
        self.assertTrue(agent1_to_agent2_success)

        low_alpha_model = SignalBasedModel(self.agent_ids, self.pos_fn, min_strength=0.01, dropout_alpha=0.1)
        mock_rng2 = Mock()
        mock_rng2.random.return_value = 0.8
        low_alpha_model.rng = mock_rng2

        low_alpha_model.update_connectivity(self.comms_matrix)
        update_calls = self.comms_matrix.update.call_args_list

        self.assertIn(call("agent1", "agent3", False), update_calls)

    def test_self_connection_handling(self):
        self.model.update_connectivity(self.comms_matrix)
        update_calls = [c[0][:2] for c in self.comms_matrix.update.call_args_list]

        for agent in self.agent_ids:
            self.assertNotIn((agent, agent), update_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
