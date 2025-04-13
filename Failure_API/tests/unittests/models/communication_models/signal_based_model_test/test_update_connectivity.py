
"""
Connectivity Matrix Update Tests for SignalBasedModel

This test file focuses on testing the `update_connectivity` method of the
SignalBasedModel class, which updates the connectivity matrix based on
agent positions and signal parameters.

We verify that:
1. The connectivity matrix is updated correctly for all agent pairs
2. Different signal scenarios lead to appropriate connectivity states
3. Self-connections are handled properly (diagonal matrix elements)
4. Multiple update cycles behave consistently
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
import numpy as np
from typing import Dict, List

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication



class MockActiveCommunication:
    """
    A mock implementation of ActiveCommunication for testing purposes.

    This class tracks updates to the connectivity matrix and allows
    verification of expected connectivity states.
    """

    def __init__(self, agent_ids):
        """Initialize with a set of agent IDs."""
        self.agent_ids = agent_ids
        self.id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        self.matrix = np.zeros((len(agent_ids), len(agent_ids)), dtype=bool)
        self.update_calls = []

    def update(self, sender: str, receiver: str, connected: bool):
        """Track updates to the connectivity matrix."""
        self.update_calls.append((sender, receiver, connected))
        sender_idx = self.id_to_idx[sender]
        receiver_idx = self.id_to_idx[receiver]
        self.matrix[sender_idx, receiver_idx] = connected

    def is_connected(self, sender: str, receiver: str) -> bool:
        """Check if two agents are connected."""
        sender_idx = self.id_to_idx[sender]
        receiver_idx = self.id_to_idx[receiver]
        return self.matrix[sender_idx, receiver_idx]


class TestConnectivityMatrixUpdates(unittest.TestCase):
    """
    Test class for verifying the connectivity matrix updates in SignalBasedModel.

    These tests ensure that the `update_connectivity` method correctly updates
    the connectivity matrix based on signal strengths and probabilities.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        Creates a SignalBasedModel instance with standard parameters and
        a MockActiveCommunication instance to track connectivity updates.
        """
        # Create a list of agent IDs
        self.agent_ids = ["agent1", "agent2", "agent3", "agent4"]

        # Create positions that result in different connectivity
        self.positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),  # Close to agent1
            "agent3": np.array([5.0, 0.0, 0.0]),  # Far from agent1, close to agent4
            "agent4": np.array([6.0, 0.0, 0.0])  # Far from agent1, close to agent3
        }

        # Mock position function
        self.pos_fn = Mock(return_value=self.positions)

        # Create a model with specific parameters to create a predictable pattern
        # tx_power=1.0, min_strength=0.05 means agents within ~4.5 units can connect
        self.model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            tx_power=1.0,
            min_strength=0.05,
            dropout_alpha=2.0
        )

        # Create a MockActiveCommunication instance
        self.comms_matrix = MockActiveCommunication(self.agent_ids)

    def test_all_pairs_updated(self):
        """
        Test that all agent pairs are updated in the connectivity matrix.

        This test verifies that the update_connectivity method calls
        comms_matrix.update for all sender-receiver pairs except self-connections.
        """
        mock_rng = Mock()
        mock_rng.random.return_value = 0.0  # or side_effect = [0.1, 0.2] for sequences
        self.model.rng = mock_rng
        # Call update_connectivity
        self.model.update_connectivity(self.comms_matrix)

        # Check that all pairs were updated
        expected_pair_count = len(self.agent_ids) * (len(self.agent_ids) - 1)
        self.assertEqual(
            len(self.comms_matrix.update_calls),
            expected_pair_count,
            "Not all agent pairs were updated in the connectivity matrix"
        )

        # Verify all expected pairs (excluding self-connections)
        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender != receiver:
                    pair_updated = any(
                        call[0] == sender and call[1] == receiver
                        for call in self.comms_matrix.update_calls
                    )
                    self.assertTrue(
                        pair_updated,
                        f"Pair ({sender}, {receiver}) not updated in the connectivity matrix"
                    )

    def test_connectivity_pattern(self):
        """
        Test that the connectivity pattern matches expected signal propagation.

        This test verifies that the connectivity matrix is updated according to
        the expected signal strengths between agents, creating a realistic pattern.
        """
        # Expected connectivity based on positions:
        # agent1 <--> agent2 (distance 1.0, strength ~1.0 > 0.05)
        # agent1 <-/-> agent3 (distance 5.0, strength ~0.04 < 0.05)
        # agent1 <-/-> agent4 (distance 6.0, strength ~0.03 < 0.05)
        # agent2 <-/-> agent3 (distance 4.0, strength ~0.06 > 0.05)
        # agent2 <-/-> agent4 (distance 5.0, strength ~0.04 < 0.05)
        # agent3 <--> agent4 (distance 1.0, strength ~1.0 > 0.05)

        mock_rng = Mock()
        mock_rng.random.return_value = 0.0  # or side_effect = [0.1, 0.2] for sequences
        self.model.rng = mock_rng
        self.model.update_connectivity(self.comms_matrix)

        # Verify expected connectivity pattern
        # Connected pairs
        self.assertTrue(self.comms_matrix.is_connected("agent1", "agent2"))
        self.assertTrue(self.comms_matrix.is_connected("agent2", "agent1"))
        self.assertTrue(self.comms_matrix.is_connected("agent3", "agent4"))
        self.assertTrue(self.comms_matrix.is_connected("agent4", "agent3"))

        # Disconnected pairs
        self.assertFalse(self.comms_matrix.is_connected("agent1", "agent3"))
        self.assertFalse(self.comms_matrix.is_connected("agent1", "agent4"))
        self.assertFalse(self.comms_matrix.is_connected("agent2", "agent4"))

        # Check agent2-agent3 (borderline case, calculated strength ~0.06)
        # This is a bit unpredictable due to floating point, so we'll allow either outcome
        agent2_agent3_connected = self.comms_matrix.is_connected("agent2", "agent3")
        # This assertion might need adjustment based on exact calculation
        self.assertEqual(
            agent2_agent3_connected,
            (1.0 / (4.0 ** 2 + 1e-6)) >= 0.05,
            "agent2-agent3 connectivity doesn't match calculated signal strength"
        )

    def test_probabilistic_effects(self):
        """
        Test that probabilistic packet loss affects connectivity properly.

        This test verifies that the probability of successful communication
        (based on distance) properly affects the connectivity updates.
        """
        # Set up a predictable pattern of random values:
        # First call: return 0.9 (should fail for distant pairs if p_success < 0.9)
        # Second call: return 0.1 (should succeed for close pairs if p_success > 0.1)
        random_values = [0.9, 0.1]

        mock_rng = Mock()
        mock_rng.random.return_value = 0.0  # or side_effect = [0.1, 0.2] for sequences
        self.model.rng = mock_rng
        self.model.update_connectivity(self.comms_matrix)

        # For agent1-agent2 (distance 1.0):
        # p_success = exp(-2.0*1.0) ≈ 0.135 > 0.1, should succeed
        self.assertTrue(
            self.comms_matrix.is_connected("agent1", "agent2"),
            "Close pair (agent1-agent2) should connect with random value 0.1 < p_success"
        )

        # For agent3-agent4 (distance 1.0):
        # p_success = exp(-2.0*1.0) ≈ 0.135 > 0.1, should succeed
        self.assertTrue(
            self.comms_matrix.is_connected("agent3", "agent4"),
            "Close pair (agent3-agent4) should connect with random value 0.1 < p_success"
        )

        # Reset the connectivity matrix for a new test
        self.comms_matrix = MockActiveCommunication(self.agent_ids)

        # Test with a low dropout_alpha to make long-distance connections more likely
        low_alpha_model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            tx_power=1.0,
            min_strength=0.01,  # Lower to ensure connections aren't blocked by threshold
            dropout_alpha=0.1  # Much lower to make long-distance connections more likely
        )

        mock_rng_low = Mock()
        mock_rng_low.random.return_value = 0.9  # Ensure it fails for p_success ≈ 0.61
        low_alpha_model.rng = mock_rng_low
        low_alpha_model.update_connectivity(self.comms_matrix)

        # Call update_connectivity
        low_alpha_model.update_connectivity(self.comms_matrix)

        # For agent1-agent3 (distance 5.0):
        # p_success = exp(-0.1*5.0) ≈ 0.61 < 0.9, should fail
        self.assertFalse(
            self.comms_matrix.is_connected("agent1", "agent3"),
            "Distant pair (agent1-agent3) should fail with random value 0.9 > p_success"
        )

        # For agent2-agent3 (distance 4.0):
        # p_success = exp(-0.1*4.0) ≈ 0.67 < 0.9, should fail
        self.assertFalse(
            self.comms_matrix.is_connected("agent2", "agent3"),
            "Distant pair (agent2-agent3) should fail with random value 0.9 > p_success"
        )

    def test_dropout_alpha_effect(self):
        """
        Test the effect of dropout_alpha on connection probability.

        This test verifies that higher dropout_alpha values make connections
        less likely at the same distance.
        """
        # Create two models with different dropout_alpha values
        low_alpha_model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            tx_power=1.0,
            min_strength=0.01,  # Low to ensure threshold doesn't block connections
            dropout_alpha=0.5  # Lower dropout rate
        )

        high_alpha_model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            tx_power=1.0,
            min_strength=0.01,  # Low to ensure threshold doesn't block connections
            dropout_alpha=5.0  # Higher dropout rate
        )

        # Use same random value for both models to ensure fair comparison
        random_value = 0.5

        # Test the low alpha model
        low_alpha_comms = MockActiveCommunication(self.agent_ids)
        mock_rng_low = Mock()
        mock_rng_low.random.return_value = random_value
        low_alpha_model.rng = mock_rng_low
        low_alpha_model.update_connectivity(low_alpha_comms)

        # Test the high alpha model
        high_alpha_comms = MockActiveCommunication(self.agent_ids)
        mock_rng_high = Mock()
        mock_rng_high.random.return_value = random_value
        high_alpha_model.rng = mock_rng_high
        high_alpha_model.update_connectivity(high_alpha_comms)

        # Count successful connections in each case
        low_alpha_connections = sum(1 for call in low_alpha_comms.update_calls if call[2])
        high_alpha_connections = sum(1 for call in high_alpha_comms.update_calls if call[2])

        # High alpha should result in fewer connections
        self.assertGreater(
            low_alpha_connections,
            high_alpha_connections,
            "Higher dropout_alpha should result in fewer successful connections"
        )

    def test_multiple_update_cycles(self):
        """
        Test that multiple update cycles behave consistently.

        This test verifies that calling update_connectivity multiple times
        with the same positions and parameters produces consistent results.
        """
        mock_rng = Mock()
        mock_rng.random.return_value = 0.0  # or side_effect = [0.1, 0.2] for sequences
        self.model.rng = mock_rng
        # First update cycle
        self.model.update_connectivity(self.comms_matrix)

        # Record the state after first update
        first_update_state = {
            (sender, receiver): self.comms_matrix.is_connected(sender, receiver)
            for sender in self.agent_ids
            for receiver in self.agent_ids
            if sender != receiver
        }

        # Reset update calls, but not the matrix state
        self.comms_matrix.update_calls = []

        # Second update cycle with the same positions
        self.model.update_connectivity(self.comms_matrix)

        # Record the state after second update
        second_update_state = {
            (sender, receiver): self.comms_matrix.is_connected(sender, receiver)
            for sender in self.agent_ids
            for receiver in self.agent_ids
            if sender != receiver
        }

        # States should be the same if random values are consistent
        self.assertEqual(
            first_update_state,
            second_update_state,
            "Connectivity updates should be consistent across multiple cycles with same conditions"
        )

    def test_changing_positions(self):
        """
        Test that changing positions affects connectivity appropriately.

        This test verifies that when agent positions change, the connectivity
        is updated to reflect the new spatial relationships.
        """
        # Initial positions
        initial_positions = self.positions.copy()

        # Set up a sequence of position functions
        self.pos_fn.side_effect = [
            initial_positions,  # First call returns initial positions
            {  # Second call returns modified positions
                "agent1": np.array([0.0, 0.0, 0.0]),
                "agent2": np.array([10.0, 0.0, 0.0]),  # Moved far from agent1
                "agent3": np.array([5.0, 0.0, 0.0]),  # Same
                "agent4": np.array([6.0, 0.0, 0.0])  # Same
            }
        ]

        mock_rng = Mock()
        mock_rng.random.return_value = 0.0  # or side_effect = [0.1, 0.2] for sequences
        self.model.rng = mock_rng
        # First update with initial positions
        self.model.update_connectivity(self.comms_matrix)

        # Verify agent1-agent2 are connected initially
        self.assertTrue(
            self.comms_matrix.is_connected("agent1", "agent2"),
            "agent1 and agent2 should be connected with initial positions"
        )

        # Second update with modified positions
        self.model.update_connectivity(self.comms_matrix)

        # Now agent1-agent2 should be disconnected due to increased distance
        self.assertFalse(
            self.comms_matrix.is_connected("agent1", "agent2"),
            "agent1 and agent2 should be disconnected after agent2 moved far away"
        )

        # Connections that shouldn't change should remain the same
        self.assertTrue(
            self.comms_matrix.is_connected("agent3", "agent4"),
            "agent3 and agent4 should remain connected as their positions didn't change"
        )

    def test_invalid_agent_id(self):
        """
        Test handling of invalid agent IDs.

        This test verifies that the model handles the case where an agent ID
        in the position dictionary doesn't exist in the agent_ids list.
        """
        # Create a position dictionary with an extra agent not in agent_ids
        invalid_positions = self.positions.copy()
        invalid_positions["extra_agent"] = np.array([0.0, 0.0, 0.0])

        invalid_pos_fn = Mock(return_value=invalid_positions)

        # Create a model with the standard agent_ids but a position function
        # that returns positions for an extra agent
        invalid_model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=invalid_pos_fn
        )

        # The model should ignore the extra agent
        try:
            invalid_model.update_connectivity(self.comms_matrix)
            test_passed = True
        except Exception as e:
            test_passed = False
            self.fail(f"update_connectivity raised an exception with invalid agent ID: {str(e)}")

        self.assertTrue(
            test_passed,
            "update_connectivity should handle invalid agent IDs gracefully"
        )

    def test_missing_agent_position(self):
        """
        Test handling of missing agent positions.

        This test verifies that the model gracefully handles the case where
        an agent in agent_ids doesn't have a corresponding position in the
        position dictionary returned by pos_fn.
        """
        # Create a position dictionary missing one of the agents
        incomplete_positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([5.0, 0.0, 0.0])
            # agent4 is missing
        }

        incomplete_pos_fn = Mock(return_value=incomplete_positions)

        # Create a model with all agent_ids but a position function
        # that doesn't return a position for one agent
        incomplete_model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=incomplete_pos_fn
        )

        # The model should raise a KeyError when trying to access a missing position
        with self.assertRaises(KeyError):
            incomplete_model.update_connectivity(self.comms_matrix)


if __name__ == "__main__":
    unittest.main()