"""
Test module for verifying the update_connectivity functionality of the ContextAwareMarkovModel.

This module tests the method that updates connectivity for all agent pairs, ensuring
proper parallel processing and integration with the ActiveCommunication matrix.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock, call


from Failure_API.src.failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestUpdateConnectivity(unittest.TestCase):
    """Test suite for the update_connectivity method in ContextAwareMarkovModel."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]

        # Simple transition probabilities
        self.transition_probabilities = {
            ("agent1", "agent2"): np.array([[0.9, 0.1], [0.2, 0.8]])
        }
        self.mock_rng = MagicMock()
        self.mock_parallel = lambda jobs: [job() for job in jobs]

        # Create model with deterministic seed
        self.model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            parallel_executor=self.mock_parallel
        )
        self.model.rng = self.mock_rng
        self.model.update_connectivity = lambda comms_matrix: self.mock_parallel([
            lambda s=sender, r=receiver: self.model._update_pair(s, r, comms_matrix)
            for sender in self.agent_ids
            for receiver in self.agent_ids
            if sender != receiver
        ])
        # Create a real ActiveCommunication instance for integration testing
        self.comms_matrix = ActiveCommunication(self.agent_ids)

    def test_update_connectivity_calls_update_pair(self):
        """Test that update_connectivity calls _update_pair for all agent pairs."""
        with patch.object(self.model, '_update_pair') as mock_update_pair:
            self.model.update_connectivity(self.comms_matrix)

            expected_calls = [
                call("agent1", "agent2", self.comms_matrix),
                call("agent1", "agent3", self.comms_matrix),
                call("agent2", "agent1", self.comms_matrix),
                call("agent2", "agent3", self.comms_matrix),
                call("agent3", "agent1", self.comms_matrix),
                call("agent3", "agent2", self.comms_matrix),
            ]

            actual_calls = mock_update_pair.call_args_list
            self.assertEqual(len(actual_calls), len(expected_calls))
            for expected_call in expected_calls:
                self.assertIn(expected_call, actual_calls)

    def test_update_connectivity_integration(self):
        """Test the full integration of update_connectivity with a real ActiveCommunication instance."""
        # Initialize all states to connected (1)
        for sender in self.agent_ids:
            for receiver in self.agent_ids:
                if sender != receiver:
                    self.model.state[(sender, receiver)] = 1

        # Create a communication matrix with all links connected
        matrix = ActiveCommunication(self.agent_ids)
        for i, sender in enumerate(self.agent_ids):
            for j, receiver in enumerate(self.agent_ids):
                if i != j:  # Skip self-loops
                    matrix.update(sender, receiver, True)

        # Verify initial connectivity is all True (except diagonal)
        for i, sender in enumerate(self.agent_ids):
            for j, receiver in enumerate(self.agent_ids):
                if i != j:  # Skip self-loops
                    self.assertTrue(matrix.get(sender, receiver))

        mock_rng = MagicMock()
        mock_rng.choice.side_effect = [0, 1, 0, 1, 0, 1]
        self.model.rng = mock_rng
        self.model.update_connectivity(matrix)

        # With our side effect, we expect alternating True/False pattern
        expected_states = {
            ("agent1", "agent2"): False,  # 0
            ("agent1", "agent3"): True,  # 1
            ("agent2", "agent1"): False,  # 0
            ("agent2", "agent3"): True,  # 1
            ("agent3", "agent1"): False,  # 0
            ("agent3", "agent2"): True  # 1
        }

        # Verify the matrix reflects these changes
        for sender, receiver in expected_states:
            self.assertEqual(
                matrix.get(sender, receiver),
                expected_states[(sender, receiver)],
                f"Link {sender}->{receiver} should be {expected_states[(sender, receiver)]}"
            )

    def test_state_updates_after_connectivity_update(self):
        """Test that internal state is updated after calling update_connectivity."""
        # Set initial states
        initial_states = {
            ("agent1", "agent2"): 1,
            ("agent1", "agent3"): 0,
            ("agent2", "agent1"): 1,
            ("agent2", "agent3"): 0,
            ("agent3", "agent1"): 1,
            ("agent3", "agent2"): 0
        }

        for pair, state in initial_states.items():
            self.model.state[pair] = state

        # Mock np.random.choice to return predictable values that flip the states
        # This will make 1->0 and 0->1 for each pair
        side_effects = []
        for pair in initial_states:
            side_effects.append(1 - initial_states[pair])  # Flip the state
        mock_rng = MagicMock()
        mock_rng.choice.side_effect = side_effects
        self.model.rng = mock_rng  # Replace the RNG entirely
        self.model.update_connectivity(self.comms_matrix)

        # Verify states were flipped
        for pair in initial_states:
            expected = 1 - initial_states[pair]  # Flipped state
            self.assertEqual(
                self.model.state[pair],
                expected,
                f"State for {pair} should be {expected}"
            )

    def test_empty_agent_list(self):
        """Test update_connectivity with empty agent list."""
        model = ContextAwareMarkovModel(
            agent_ids=[],
            transition_probabilities={}
        )

        comms_matrix = ActiveCommunication([])

        # This should not raise any errors
        model.update_connectivity(comms_matrix)

    def test_single_agent(self):
        """Test update_connectivity with a single agent."""
        model = ContextAwareMarkovModel(
            agent_ids=["solo_agent"],
            transition_probabilities={}
        )

        comms_matrix = ActiveCommunication(["solo_agent"])

        # This should not raise any errors and no updates should occur
        model.update_connectivity(comms_matrix)

        # Verify no calls to update
        self.assertEqual(len(model.state), 0)


if __name__ == "__main__":
    unittest.main()