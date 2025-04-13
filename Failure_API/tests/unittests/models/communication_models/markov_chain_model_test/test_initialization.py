"""
Test module for verifying the initialization behavior of the ContextAwareMarkovModel class.

This module focuses on testing the initialization process, including parameter handling,
state initialization, and proper storage of agent IDs and transition probabilities.
"""

import unittest
import numpy as np
from collections import defaultdict
from unittest.mock import patch, MagicMock

from Failure_API.src.failure_api.communication_models.markov_chain_based import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels




class TestContextAwareMarkovModelInitialization(unittest.TestCase):
    """Test suite for initialization of the ContextAwareMarkovModel class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]

        # Define transition probabilities for each link
        # Format: {(sender, receiver): 2x2 matrix}
        self.transition_probabilities = {
            ("agent1", "agent2"): np.array([[0.9, 0.1], [0.2, 0.8]]),
            ("agent1", "agent3"): np.array([[0.8, 0.2], [0.3, 0.7]]),
            ("agent2", "agent1"): np.array([[0.7, 0.3], [0.4, 0.6]]),
            ("agent2", "agent3"): np.array([[0.6, 0.4], [0.5, 0.5]]),
            ("agent3", "agent1"): np.array([[0.5, 0.5], [0.6, 0.4]]),
            ("agent3", "agent2"): np.array([[0.4, 0.6], [0.7, 0.3]]),
        }

        # Mock context, time, and traffic functions
        self.mock_context_fn = lambda: {"weather": 0.3}
        self.mock_time_fn = lambda: 5
        self.mock_traffic_fn = lambda: {("agent1", "agent2"): 0.4}

    def test_basic_initialization(self):
        """Test initialization with minimal required parameters."""
        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities
        )

        # Verify agent_ids are stored correctly
        self.assertEqual(model.agent_ids, self.agent_ids)

        # Verify transition_probabilities are stored correctly
        self.assertEqual(model.transition_probabilities, self.transition_probabilities)

        # Verify optional parameters are None
        self.assertIsNone(model.context_fn)
        self.assertIsNone(model.time_fn)
        self.assertIsNone(model.traffic_fn)

        # Verify state is a defaultdict initialized to 1 (connected)
        self.assertIsInstance(model.state, defaultdict)
        self.assertEqual(model.state[("agent1", "agent2")], 1)
        self.assertEqual(model.state[("nonexistent", "agent")], 1)  # Test defaultdict behavior

    def test_full_initialization(self):
        """Test initialization with all parameters provided."""
        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            context_fn=self.mock_context_fn,
            time_fn=self.mock_time_fn,
            traffic_fn=self.mock_traffic_fn
        )

        # Verify all parameters are stored correctly
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertEqual(model.transition_probabilities, self.transition_probabilities)
        self.assertEqual(model.context_fn(), {"weather": 0.3})
        self.assertEqual(model.time_fn(), 5)
        self.assertEqual(model.traffic_fn(), {("agent1", "agent2"): 0.4})

    def test_empty_agent_list(self):
        """Test initialization with an empty agent list."""
        model = ContextAwareMarkovModel(
            agent_ids=[],
            transition_probabilities={}
        )

        self.assertEqual(model.agent_ids, [])
        self.assertEqual(model.transition_probabilities, {})

    def test_single_agent(self):
        """Test initialization with a single agent (no communication possible)."""
        model = ContextAwareMarkovModel(
            agent_ids=["solo_agent"],
            transition_probabilities={}
        )

        self.assertEqual(model.agent_ids, ["solo_agent"])

        # Create a communication matrix and verify no updates happen
        comms_matrix = ActiveCommunication(["solo_agent"])
        model.update_connectivity(comms_matrix)

        # No links should be updated since there's only one agent
        # This tests that the model doesn't crash with a single agent

    def test_create_initial_matrix(self):
        """Test the static method to create initial connectivity matrix."""
        agent_ids = ["agent1", "agent2", "agent3"]
        matrix = ContextAwareMarkovModel.create_initial_matrix(agent_ids)

        # Verify matrix shape
        self.assertEqual(matrix.shape, (3, 3))

        # Verify diagonal elements are False (no self-communication)
        for i in range(3):
            self.assertFalse(matrix[i, i])

        # Verify all other elements are True (all other links possible)
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertTrue(matrix[i, j])

    def test_inheritance(self):
        """Test that the model inherits from CommunicationModels base class."""
        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities
        )

        from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels
        self.assertIsInstance(model, CommunicationModels)


if __name__ == "__main__":
    unittest.main()