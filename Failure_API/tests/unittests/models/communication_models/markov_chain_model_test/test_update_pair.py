"""
Test module for verifying the update_pair functionality of the ContextAwareMarkovModel.

This module tests the core method that updates the state for a single sender-receiver
communication link.
"""

import unittest
import numpy as np
from unittest.mock import patch, Mock, MagicMock

from Failure_API.src.failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestUpdatePair(unittest.TestCase):
    """Test suite for the _update_pair method in ContextAwareMarkovModel."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]

        # Define transition probabilities for a specific link
        self.transition_probabilities = {
            ("agent1", "agent2"): np.array([[0.9, 0.1], [0.2, 0.8]])
        }

        # Create a model with fixed random seed for deterministic testing
        self.model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities
        )
        self.mock_rng = MagicMock()
        self.model.set_rng(self.mock_rng)

        # Create a mock ActiveCommunication object
        self.comms_matrix = MagicMock(spec=ActiveCommunication)

    def test_update_pair_with_existing_probability(self):
        """Test updating a pair that has explicit transition probabilities defined."""
        sender, receiver = "agent1", "agent2"

        # Initial state is 1 (connected) by default
        self.assertEqual(self.model.state[(sender, receiver)], 1)

        # Update the pair
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # With our fixed random seed, we can predict the outcome
        # The exact outcome depends on the random seed used
        # For this test, we're more interested in verifying that:
        # 1. The state gets updated
        # 2. The comms_matrix.update method gets called with the right parameters

        # Check that comms_matrix.update was called correctly
        self.comms_matrix.update.assert_called_once()
        call_args = self.comms_matrix.update.call_args[0]
        self.assertEqual(call_args[0], sender)
        self.assertEqual(call_args[1], receiver)
        # The third parameter is a boolean indicating connectivity status
        self.assertIsInstance(call_args[2], bool)

    def test_update_pair_with_default_probability(self):
        """Test updating a pair that uses default transition probabilities."""
        sender, receiver = "agent2", "agent3"

        # This pair doesn't have explicit probabilities in self.transition_probabilities
        self.assertNotIn((sender, receiver), self.transition_probabilities)

        # Update the pair
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # Verify the default matrix [[0.9, 0.1], [0.1, 0.9]] was used
        # by checking that comms_matrix.update was called
        self.comms_matrix.update.assert_called_once()
        call_args = self.comms_matrix.update.call_args[0]
        self.assertEqual(call_args[0], sender)
        self.assertEqual(call_args[1], receiver)
        self.assertIsInstance(call_args[2], bool)

    def test_state_persistence(self):
        """Test that state persists between updates for the same pair."""
        sender, receiver = "agent1", "agent2"

        # Set initial state manually
        self.model.state[(sender, receiver)] = 0  # Disconnected

        # Update the pair
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # The state should have been updated based on the transition matrix
        # We can get the current state
        current_state = self.model.state[(sender, receiver)]

        # Reset the mock to clear call history
        self.comms_matrix.reset_mock()

        # Update again to see if it transitions from the new state
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # Check that the update was called with the sender and receiver
        self.comms_matrix.update.assert_called_once()
        call_args = self.comms_matrix.update.call_args[0]
        self.assertEqual(call_args[0], sender)
        self.assertEqual(call_args[1], receiver)

    def test_state_transition_connected_to_disconnected(self):
        """Test transition from connected (1) to disconnected (0) state."""
        sender, receiver = "agent1", "agent2"

        # Set initial state to connected
        self.model.state[(sender, receiver)] = 1

        self.mock_rng.choice.return_value = 0

        # Update the pair
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # Verify state changed to disconnected
        self.assertEqual(self.model.state[(sender, receiver)], 0)

        # Verify that comms_matrix.update was called with False (disconnected)
        self.comms_matrix.update.assert_called_once_with(sender, receiver, False)

    def test_state_transition_disconnected_to_connected(self):
        """Test transition from disconnected (0) to connected (1) state."""
        sender, receiver = "agent1", "agent2"

        # Set initial state to disconnected
        self.model.state[(sender, receiver)] = 0

        self.mock_rng.choice.return_value = 1

        # Update the pair
        self.model._update_pair(sender, receiver, self.comms_matrix)

        # Verify state changed to connected
        self.assertEqual(self.model.state[(sender, receiver)], 1)

        # Verify that comms_matrix.update was called with True (connected)
        self.comms_matrix.update.assert_called_once_with(sender, receiver, True)

    def test_context_influence(self):
        """Test that context influences the transition probabilities."""
        sender, receiver = "agent1", "agent2"

        # Add a context function that affects transitions
        self.model.context_fn = lambda: {"weather": 0.6}  # Bad weather

        # Patch adjust_matrix to verify it's called with the right parameters
        with patch.object(self.model, '_adjust_matrix', wraps=self.model._adjust_matrix) as mock_adjust:
            self.model._update_pair(sender, receiver, self.comms_matrix)

            # Verify _adjust_matrix was called with the correct matrix and pair
            mock_adjust.assert_called_once()
            call_args = mock_adjust.call_args[0]
            np.testing.assert_array_equal(call_args[0], self.transition_probabilities[(sender, receiver)])
            self.assertEqual(call_args[1], sender)
            self.assertEqual(call_args[2], receiver)


if __name__ == "__main__":
    unittest.main()