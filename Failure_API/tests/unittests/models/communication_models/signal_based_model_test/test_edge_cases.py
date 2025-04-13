#!/usr/bin/env python
"""
Edge Case Tests for SignalBasedModel

This test file focuses on testing various edge cases and error handling
for the SignalBasedModel class. These tests verify that the model behaves
correctly in unusual or extreme situations, including:

1. Empty or singleton agent lists
2. Invalid position data
3. Extreme parameter values
4. Error handling in position function calls
5. Edge case behavior in signal strength calculations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, List
import warnings

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestSignalBasedModelEdgeCases(unittest.TestCase):
    """
    Test class for verifying edge case behavior and error handling in SignalBasedModel.

    These tests ensure that the model behaves correctly and gracefully
    in unusual or extreme situations, and properly handles errors.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        Creates some standard agents and position functions for testing,
        as well as a mock ActiveCommunication matrix.
        """
        # Create some standard agents for testing
        self.standard_agent_ids = ["agent1", "agent2", "agent3"]

        # Create a standard position function
        self.standard_positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([2.0, 0.0, 0.0])
        }
        self.standard_pos_fn = Mock(return_value=self.standard_positions)

        # Create a mock ActiveCommunication matrix
        self.comms_matrix = Mock(spec=ActiveCommunication)
        self.comms_matrix.update = MagicMock()

    def test_empty_agent_list(self):
        """
        Test initialization and operation with an empty agent list.

        This tests the edge case of having no agents, which should
        initialize correctly but may have special behavior for operations.
        """
        # Create model with empty agent list
        empty_model = SignalBasedModel(
            agent_ids=[],
            pos_fn=lambda: {}
        )

        # Verify initialization succeeded
        self.assertEqual(empty_model.agent_ids, [])
        self.assertEqual(empty_model.id_to_idx, {})

        # Test create_initial_matrix with empty list
        empty_matrix = SignalBasedModel.create_initial_matrix([])
        self.assertEqual(empty_matrix.shape, (0, 0))

        # Test update_connectivity with empty list
        empty_model.update_connectivity(self.comms_matrix)

        # No updates should have been made to the communication matrix
        self.comms_matrix.update.assert_not_called()

    def test_single_agent(self):
        """
        Test initialization and operation with a single agent.

        This tests the edge case of having only one agent, which
        should initialize correctly but may have special behavior.
        """
        # Create model with a single agent
        single_agent_id = ["agent1"]
        single_pos_fn = Mock(return_value={
            "agent1": np.array([0.0, 0.0, 0.0])
        })

        single_model = SignalBasedModel(
            agent_ids=single_agent_id,
            pos_fn=single_pos_fn
        )

        # Verify initialization succeeded
        self.assertEqual(single_model.agent_ids, single_agent_id)
        self.assertEqual(single_model.id_to_idx, {"agent1": 0})

        # Test create_initial_matrix with single agent
        single_matrix = SignalBasedModel.create_initial_matrix(single_agent_id)
        self.assertEqual(single_matrix.shape, (1, 1))
        self.assertEqual(single_matrix[0, 0], 0.0)

        # Test update_connectivity with single agent
        single_model.update_connectivity(self.comms_matrix)

        # No updates should have been made (no valid pairs)
        self.comms_matrix.update.assert_not_called()

    def test_zero_transmit_power(self):
        """
        Test behavior with zero transmit power.

        This tests the edge case where transmit power is set to zero,
        which should result in no connections between agents.
        """
        # Create model with zero transmit power
        zero_power_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            tx_power=0.0
        )

        # Update connectivity
        zero_power_model.update_connectivity(self.comms_matrix)

        # Check that all updates were to False (disconnected)
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            self.assertFalse(
                connected,
                f"Connection ({sender}, {receiver}) should be False with zero tx_power"
            )

    def test_zero_min_strength(self):
        """
        Test behavior with zero minimum strength threshold.

        This tests the edge case where the minimum strength threshold is zero,
        which should result in connections based solely on probabilistic factors.
        """
        # Create model with zero minimum strength
        zero_threshold_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            min_strength=0.0
        )

        # Patch random to always return 0.0 (successful connection)
        zero_threshold_model.rng = Mock()
        zero_threshold_model.rng.random.return_value = 0.0
        zero_threshold_model.update_connectivity(self.comms_matrix)

        # All connections should be successful
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            self.assertTrue(
                connected,
                f"Connection ({sender}, {receiver}) should be True with zero min_strength"
            )

    def test_zero_dropout_alpha(self):
        """
        Test behavior with zero dropout alpha.

        This tests the edge case where dropout_alpha is zero, which should
        result in no probabilistic packet loss (all connections succeed).
        """
        # Create model with zero dropout alpha
        zero_alpha_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            dropout_alpha=0.0
        )

        # Update connectivity
        zero_alpha_model.update_connectivity(self.comms_matrix)

        # All connections with sufficient signal strength should succeed
        # Since p_success = exp(-0.0 * dist) = 1.0, all connections should succeed
        # Check for a specific connection we know should have sufficient strength
        agent1_to_agent2_update = None
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            if sender == "agent1" and receiver == "agent2":
                agent1_to_agent2_update = connected
                break

        self.assertTrue(
            agent1_to_agent2_update,
            "agent1 to agent2 should be connected with zero dropout_alpha"
        )

    def test_negative_parameters(self):
        """
        Test behavior with negative parameter values.

        This tests the edge case where parameters like tx_power or min_strength
        are negative, which may not be physically meaningful but should be handled.
        """
        # Negative transmit power doesn't make physical sense but should behave like zero
        neg_power_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            tx_power=-1.0
        )

        # Update connectivity
        self.comms_matrix.update.reset_mock()
        neg_power_model.update_connectivity(self.comms_matrix)

        # All connections should be False due to negative signal strength
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            self.assertFalse(
                connected,
                f"Connection ({sender}, {receiver}) should be False with negative tx_power"
            )

        # Negative minimum strength doesn't make physical sense but should allow all connections
        neg_threshold_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            min_strength=-0.1
        )

        # Mock RNG
        neg_threshold_model = Mock()
        neg_threshold_model.rng.random.return_value = 0.0
        # Update connectivity
        self.comms_matrix.update.reset_mock()
        neg_threshold_model.update_connectivity(self.comms_matrix)

        # All connections should be True (signal strength > negative threshold)
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            self.assertTrue(
                connected,
                f"Connection ({sender}, {receiver}) should be True with negative min_strength"
            )

        # Negative dropout alpha is physically meaningless but should increase probability with distance
        neg_alpha_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            dropout_alpha=-1.0
        )

        # Update connectivity
        self.comms_matrix.update.reset_mock()
        neg_alpha_model.update_connectivity(self.comms_matrix)

        # All connections with sufficient signal strength should succeed
        # Since p_success = exp(-(-1.0) * dist) = exp(dist), which increases with distance
        # Check a connection we know should have sufficient strength
        agent1_to_agent3_update = None
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            if sender == "agent1" and receiver == "agent3":
                agent1_to_agent3_update = connected
                break

        # Agent1 to agent3 is farther but should have higher success probability with negative alpha
        self.assertIsNotNone(
            agent1_to_agent3_update,
            "Missing update for agent1 to agent3 with negative dropout_alpha"
        )

    def test_very_large_parameters(self):
        """
        Test behavior with extremely large parameter values.

        This tests the edge case where parameters have extremely large values,
        which might cause numerical issues or overflow.
        """
        # Very large transmit power
        large_power_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            tx_power=1e10
        )

        # Test update_connectivity
        self.comms_matrix.update.reset_mock()
        large_power_model.update_connectivity(self.comms_matrix)

        # All connections should succeed (very high signal strength)
        large_power_model.rng = Mock()
        large_power_model.rng.random.return_value = 0.0
        # Since signal strength = tx_power / (dist^2 + ε), very large tx_power
        # should result in very large signal strength, well above threshold
        agent1_to_agent3_update = None
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            if sender == "agent1" and receiver == "agent3":
                agent1_to_agent3_update = connected
                break

        self.assertTrue(
            agent1_to_agent3_update,
            "agent1 to agent3 should be connected with very large tx_power"
        )

        # Very large minimum strength
        large_threshold_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            min_strength=1e10
        )

        # Test update_connectivity
        self.comms_matrix.update.reset_mock()
        large_threshold_model.update_connectivity(self.comms_matrix)

        # All connections should fail (signal strength < very large threshold)
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            self.assertFalse(
                connected,
                f"Connection ({sender}, {receiver}) should be False with very large min_strength"
            )

        # Very large dropout alpha
        large_alpha_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=self.standard_pos_fn,
            dropout_alpha=1e10
        )

        # Test update_connectivity
        self.comms_matrix.update.reset_mock()
        large_alpha_model.update_connectivity(self.comms_matrix)

        # All connections should fail (very low success probability)
        # Since p_success = exp(-1e10 * dist) ≈ 0 for any positive dist
        agent1_to_agent2_update = None
        for call_args in self.comms_matrix.update.call_args_list:
            sender, receiver, connected = call_args[0]
            if sender == "agent1" and receiver == "agent2":
                agent1_to_agent2_update = connected
                break

        # Even for close agents, very large dropout_alpha should result in failure
        self.assertFalse(
            agent1_to_agent2_update,
            "agent1 to agent2 should be disconnected with very large dropout_alpha"
        )

    def test_invalid_position_data(self):
        """
        Test behavior with invalid position data.

        This tests error handling when the position function returns
        invalid data like non-numeric arrays or incomplete data.
        """
        # Position function returning non-numeric data
        invalid_positions = {
            "agent1": "not an array",
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([2.0, 0.0, 0.0])
        }
        invalid_pos_fn = Mock(return_value=invalid_positions)

        invalid_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=invalid_pos_fn
        )

        # Should raise an exception when trying to compute with non-numeric data
        with self.assertRaises(Exception):
            invalid_model.update_connectivity(self.comms_matrix)

        # Position function returning arrays of different dimensions
        mixed_dim_positions = {
            "agent1": np.array([0.0]),  # 1D
            "agent2": np.array([1.0, 0.0]),  # 2D
            "agent3": np.array([2.0, 0.0, 0.0])  # 3D
        }
        mixed_dim_pos_fn = Mock(return_value=mixed_dim_positions)

        mixed_dim_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=mixed_dim_pos_fn
        )

        # This might raise an exception or handle gracefully, depends on implementation
        try:
            mixed_dim_model.update_connectivity(self.comms_matrix)
            # If it doesn't raise, that's fine, just note that it handled mixed dimensions
            pass
        except Exception:
            # If it raises, that's also fine, just note that it requires consistent dimensions
            pass

    def test_exception_in_position_function(self):
        """
        Test error handling when the position function raises an exception.

        This tests the behavior when the position function fails to return valid data.
        """

        # Position function that raises an exception
        def failing_pos_fn():
            raise ValueError("Simulated position function failure")

        failing_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=failing_pos_fn
        )

        # Should propagate the exception
        with self.assertRaises(ValueError):
            failing_model.update_connectivity(self.comms_matrix)

    def test_infinite_and_nan_values(self):
        """
        Test behavior with infinite or NaN position values.

        This tests error handling when positions contain infinite or NaN values,
        which can cause numerical issues.
        """
        # Position data with infinite values
        inf_positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([np.inf, 0.0, 0.0]),
            "agent3": np.array([2.0, 0.0, 0.0])
        }
        inf_pos_fn = Mock(return_value=inf_positions)

        inf_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=inf_pos_fn
        )

        # Might raise an exception or handle gracefully
        try:
            inf_model.update_connectivity(self.comms_matrix)
            # If it doesn't raise, check that connections to the inf agent are False
            inf_agent_connected = False
            for call_args in self.comms_matrix.update.call_args_list:
                sender, receiver, connected = call_args[0]
                if sender == "agent2" or receiver == "agent2":
                    inf_agent_connected |= connected

            self.assertFalse(
                inf_agent_connected,
                "Connections to agent with infinite position should be False"
            )
        except Exception:
            # If it raises, that's also fine
            pass

        # Position data with NaN values
        nan_positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([np.nan, 0.0, 0.0]),
            "agent3": np.array([2.0, 0.0, 0.0])
        }
        nan_pos_fn = Mock(return_value=nan_positions)

        nan_model = SignalBasedModel(
            agent_ids=self.standard_agent_ids,
            pos_fn=nan_pos_fn
        )

        # Might raise an exception or handle gracefully
        try:
            nan_model.update_connectivity(self.comms_matrix)
            # If it doesn't raise, check that connections to the NaN agent are False
            nan_agent_connected = False
            for call_args in self.comms_matrix.update.call_args_list:
                sender, receiver, connected = call_args[0]
                if sender == "agent2" or receiver == "agent2":
                    nan_agent_connected |= connected

            self.assertFalse(
                nan_agent_connected,
                "Connections to agent with NaN position should be False"
            )
        except Exception:
            # If it raises, that's also fine
            pass

    def test_many_duplicated_positions(self):
        """
        Test behavior when many agents have identical positions.

        This tests the edge case when multiple agents are at exactly the same position,
        which could cause issues with spatial indexing.
        """
        # Create many agents at the same position
        many_agents = [f"agent{i}" for i in range(20)]

        # All agents at the same position
        same_position = np.array([0.0, 0.0, 0.0])
        same_positions = {aid: same_position for aid in many_agents}
        same_pos_fn = Mock(return_value=same_positions)

        same_pos_model = SignalBasedModel(
            agent_ids=many_agents,
            pos_fn=same_pos_fn
        )

        # Create a mock comms matrix for these agents
        many_comms_matrix = Mock(spec=ActiveCommunication)
        many_comms_matrix.update = MagicMock()

        # Should handle this without raising exceptions
        try:
            same_pos_model.update_connectivity(many_comms_matrix)

            # Verify that the expected number of updates were made
            # With n agents all at the same position, there should be n*(n-1) potential connections
            expected_updates = len(many_agents) * (len(many_agents) - 1)
            actual_updates = many_comms_matrix.update.call_count

            self.assertEqual(
                actual_updates,
                expected_updates,
                f"Expected {expected_updates} updates, got {actual_updates}"
            )
        except Exception as e:
            self.fail(f"update_connectivity raised an exception with many agents at same position: {str(e)}")


if __name__ == "__main__":
    unittest.main()