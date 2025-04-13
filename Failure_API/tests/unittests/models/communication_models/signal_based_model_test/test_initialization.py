
"""
Initialization Tests for SignalBasedModel

This test file focuses on testing the initialization of the SignalBasedModel class.
We verify that:
1. The constructor properly sets all instance variables
2. Default parameters work as expected
3. Agent IDs are correctly stored and indexed
4. The position function is properly integrated

These tests form the foundation for ensuring the model is correctly set up
before testing more complex behaviors.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, List

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestSignalBasedModelInitialization(unittest.TestCase):
    """
    Test class for verifying the initialization of SignalBasedModel.

    These tests ensure that the model is properly configured during construction
    with various parameter combinations and edge cases.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        This method creates:
        1. A list of agent IDs for testing
        2. A mock position function that returns predetermined positions
        """
        # Create a list of agent IDs for testing
        self.agent_ids = ["agent1", "agent2", "agent3"]

        # Create a mock position function
        self.positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([0.0, 1.0, 0.0])
        }

        # Mock position function
        self.pos_fn = Mock(return_value=self.positions)

    def test_initialization_with_default_parameters(self):
        """
        Test that the model initializes correctly with default parameters.

        This test verifies that when only required parameters are provided,
        the default values for optional parameters are correctly set.
        """
        # Create model with default parameters
        model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn
        )

        # Check that required parameters are set correctly
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertEqual(model.pos_fn, self.pos_fn)

        # Check that default parameters are set correctly
        self.assertEqual(model.tx_power, 15.0)
        self.assertEqual(model.min_strength, 0.01)
        self.assertEqual(model.dropout_alpha, 0.2)

        # Verify id_to_idx mapping is correct
        expected_id_to_idx = {
            "agent1": 0,
            "agent2": 1,
            "agent3": 2
        }
        self.assertEqual(model.id_to_idx, expected_id_to_idx)

    def test_initialization_with_custom_parameters(self):
        """
        Test that the model initializes correctly with custom parameters.

        This test ensures that when custom values are provided for optional
        parameters, they override the defaults and are correctly set.
        """
        # Custom parameter values
        tx_power = 2.5
        min_strength = 0.1
        dropout_alpha = 1.5

        # Create model with custom parameters
        model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            tx_power=tx_power,
            min_strength=min_strength,
            dropout_alpha=dropout_alpha
        )

        # Check that all parameters are set correctly
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertEqual(model.pos_fn, self.pos_fn)
        self.assertEqual(model.tx_power, tx_power)
        self.assertEqual(model.min_strength, min_strength)
        self.assertEqual(model.dropout_alpha, dropout_alpha)

    def test_empty_agent_list(self):
        """
        Test initialization with an empty agent list.

        This tests the edge case of initializing the model without any agents,
        which could potentially cause issues in the connectivity matrix creation.
        """
        # Create model with empty agent list
        model = SignalBasedModel(
            agent_ids=[],
            pos_fn=lambda: {}
        )

        # Check that agent_ids is an empty list
        self.assertEqual(model.agent_ids, [])

        # Check that id_to_idx is an empty dictionary
        self.assertEqual(model.id_to_idx, {})

    def test_single_agent(self):
        """
        Test initialization with a single agent.

        This tests the edge case of having only one agent, which should
        still work but might have special implications for connectivity.
        """
        # Single agent ID
        single_agent_id = ["agent1"]

        # Single position
        single_pos_fn = Mock(return_value={"agent1": np.array([0.0, 0.0, 0.0])})

        # Create model with single agent
        model = SignalBasedModel(
            agent_ids=single_agent_id,
            pos_fn=single_pos_fn
        )

        # Check that agent_ids contains just the one agent
        self.assertEqual(model.agent_ids, single_agent_id)

        # Check that id_to_idx maps the single agent to index 0
        self.assertEqual(model.id_to_idx, {"agent1": 0})

    def test_position_function_called(self):
        """
        Test that the position function is integrated correctly.

        This verifies that the position function is not called during initialization
        but can be called later to retrieve positions.
        """
        # Create model
        model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn
        )

        # Verify that pos_fn was not called during initialization
        self.pos_fn.assert_not_called()

        # Create a mock ActiveCommunication matrix
        comms_matrix = Mock(spec=ActiveCommunication)

        # Call update_connectivity, which should call pos_fn
        model.update_connectivity(comms_matrix)

        # Verify that pos_fn was called exactly once
        self.pos_fn.assert_called_once()

    def test_invalid_parameter_types(self):
        """
        Test initialization with invalid parameter types.

        This test ensures that appropriate type errors are raised when
        parameters of the wrong type are provided.
        """
        # Test with non-list agent_ids
        with self.assertRaises(TypeError):
            SignalBasedModel(
                agent_ids="not_a_list",  # Should be a list
                pos_fn=self.pos_fn
            )

        # Test with non-callable pos_fn
        with self.assertRaises(TypeError):
            SignalBasedModel(
                agent_ids=self.agent_ids,
                pos_fn="not_a_function"  # Should be callable
            )

        # Test with non-float tx_power
        with self.assertRaises(TypeError):
            SignalBasedModel(
                agent_ids=self.agent_ids,
                pos_fn=self.pos_fn,
                tx_power="not_a_float"  # Should be a float
            )

    def test_create_initial_matrix(self):
        """
        Test the create_initial_matrix static method.

        This verifies that the method correctly creates a matrix of ones
        with the appropriate dimensions based on the number of agents.
        """
        # Test with our test agent IDs
        matrix = SignalBasedModel.create_initial_matrix(self.agent_ids)

        # Check matrix dimensions
        self.assertEqual(matrix.shape, (3, 3))

        # Check that all values are 1.0
        self.assertTrue(np.all(np.diag(matrix)  == 0.0))

        # Off-diagonal should be 1.0
        off_diag = matrix[~np.eye(len(self.agent_ids), dtype=bool)]
        self.assertTrue(np.all(off_diag == 1.0), "All off-diagonal values should be 1.0")

        # Test with empty agent list
        empty_matrix = SignalBasedModel.create_initial_matrix([])

        # Check empty matrix dimensions
        self.assertEqual(empty_matrix.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()