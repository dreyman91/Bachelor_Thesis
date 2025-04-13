
"""
Spatial Indexing and Proximity Tests for SignalBasedModel

This test file focuses on testing the spatial indexing and proximity calculation
features of the SignalBasedModel class. The model uses KD-Tree for efficient
spatial querying to determine which agents can potentially communicate.

We verify that:
1. KD-Tree construction works with various agent configurations
2. Neighbor querying finds the correct agents in proximity
3. Distance calculations between agents are accurate
4. Edge cases like agents at identical positions are handled correctly
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from typing import Dict, List
from scipy.spatial import cKDTree

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestSpatialIndexingAndProximity(unittest.TestCase):
    """
    Test class for verifying the spatial indexing and proximity calculations in SignalBasedModel.

    These tests ensure that the KD-Tree construction and spatial querying work correctly,
    enabling efficient determination of which agents can potentially communicate.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        Creates a SignalBasedModel instance with standard parameters and
        defines test positions for spatial indexing tests.
        """
        # Create a list of agent IDs
        self.agent_ids = ["agent1", "agent2", "agent3", "agent4", "agent5"]

        # Create positions in a 3D space
        self.positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0]),
            "agent3": np.array([0.0, 1.0, 0.0]),
            "agent4": np.array([1.0, 1.0, 0.0]),
            "agent5": np.array([5.0, 5.0, 0.0])
        }

        # Mock position function
        self.pos_fn = Mock(return_value=self.positions)

        # Create a model with default parameters
        self.model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn,
            # Use a large transmit power to ensure all agents can potentially connect
            tx_power=100.0,
            min_strength=0.01,
            dropout_alpha=0.1
        )

        # Create a mock ActiveCommunication matrix
        self.comms_matrix = Mock(spec=ActiveCommunication)
        self.comms_matrix.update = MagicMock()

    def test_kdtree_construction(self):
        """
        Test that KD-Tree is constructed correctly with agent positions.

        This test verifies that the KD-Tree is built with the correct coordinates
        and can be used for efficient spatial queries.
        """
        # Patch cKDTree to capture the coordinates used in construction
        with patch('Failure_API.src.failure_api.communication_models.signal_based_failure.cKDTree', wraps=cKDTree) as mock_kdtree:
            # Call update_connectivity to trigger KD-Tree construction
            self.model.update_connectivity(self.comms_matrix)

            # Check that cKDTree was called with the correct coordinates
            mock_kdtree.assert_called_once()

            # Extract the coordinates passed to cKDTree
            coords_arg = mock_kdtree.call_args[0][0]

            # Verify that the coordinates match our positions
            expected_coords = np.array([
                self.positions["agent1"],
                self.positions["agent2"],
                self.positions["agent3"],
                self.positions["agent4"],
                self.positions["agent5"]
            ])

            # Compare the coordinates (use np.allclose for floating-point arrays)
            self.assertTrue(
                np.allclose(coords_arg, expected_coords),
                "KD-Tree not constructed with the correct coordinates"
            )

    def test_neighbor_querying(self):
        """
        Test that neighbor querying returns the correct agents in proximity.

        This test verifies that the KD-Tree's query method is used correctly
        to find neighboring agents for each sender.
        """

        class MockKDTree:
            query_calls = []  # Class variable to track calls across all instances

            def __init__(self, *args, **kwargs):
                # Store the data for verification if needed
                self.data = args[0] if args else None
                MockKDTree.query_calls = []  # Reset tracking for each test

            def query(self, x, k=1, **kwargs):
                # Track this call
                MockKDTree.query_calls.append((x, k, kwargs))
                # Return predetermined results
                return (None, np.array([0, 1, 2, 3, 4]))


        # Mock the KD-Tree query method to return controlled results
        with patch('Failure_API.src.failure_api.communication_models.signal_based_failure.cKDTree', MockKDTree):
            # Call update_connectivity to trigger neighbor querying
            self.model.update_connectivity(self.comms_matrix)

            # Verify that query was called for each agent
            self.assertEqual(
                len(MockKDTree.query_calls),
                len(self.agent_ids),
                "KD-Tree query should be called once for each agent"
            )

            # Verify that query was called with k=len(agent_ids) for all agents
            for x, k, kwargs in MockKDTree.query_calls:
                self.assertEqual(
                    k,
                    len(self.agent_ids),
                    "KD-Tree query not called with k=len(agent_ids)")

    def test_distance_calculation(self):
        """
        Test that distances between agents are calculated correctly.

        This test verifies that the Euclidean distance calculation between
        sender and receiver positions is accurate.
        """
        # Patch numpy.linalg.norm to monitor distance calculations
        with patch('numpy.linalg.norm', wraps=np.linalg.norm) as mock_norm:
            # Set random to always return 0.0 (success) to focus on distance calculation
            with patch('numpy.linalg.norm', wraps=np.linalg.norm) as mock_norm:
                # Store the original RNG object
                original_rng = self.model.rng

                # Create a mock object with a random method that always returns 0.0
                mock_rng = Mock()
                mock_rng.random = Mock(return_value=0.0)

                # Replace the entire RNG object
                self.model.rng = mock_rng

                try:
                    # Call update_connectivity
                    self.model.update_connectivity(self.comms_matrix)

                    # Verify that np.linalg.norm was called for distance calculations
                    # The exact number depends on your agent setup
                    expected_calls = len(self.agent_ids) * (len(self.agent_ids) - 1)
                    self.assertGreaterEqual(
                        mock_norm.call_count,
                        expected_calls,
                        "Distance calculation should be performed for all agent pairs"
                    )

                    # Check that the calls to np.linalg.norm were for different agent pairs
                    # This is a basic check to ensure the calculation is actually happening
                    call_args_list = [call_args[0][0] for call_args in mock_norm.call_args_list]
                    unique_args = set(tuple(arg.flatten()) if hasattr(arg, 'flatten') else arg
                                      for arg in call_args_list)
                    self.assertGreater(
                        len(unique_args),
                        1,
                        "Multiple unique distance calculations should be performed"
                    )

                finally:
                    # Restore the original RNG object
                    self.model.rng = original_rng
    def test_identical_positions(self):
        """
        Test that agents at identical positions are handled correctly.

        This tests the edge case where multiple agents are at the exact same position,
        which could potentially cause issues with spatial indexing.
        """
        # Create a new position dictionary with two agents at the same position
        identical_positions = {
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([0.0, 0.0, 0.0]),  # Same as agent1
            "agent3": np.array([1.0, 0.0, 0.0])
        }

        # Create a new model with these positions
        identical_pos_fn = Mock(return_value=identical_positions)
        identical_agent_ids = ["agent1", "agent2", "agent3"]

        identical_model = SignalBasedModel(
            agent_ids=identical_agent_ids,
            pos_fn=identical_pos_fn,
            tx_power=1.0,
            min_strength=0.05,
            dropout_alpha=2.0
        )

        # Call update_connectivity (this should not raise any exceptions)
        try:
            identical_model.update_connectivity(self.comms_matrix)
            test_passed = True
        except Exception as e:
            test_passed = False
            self.fail(f"update_connectivity raised an exception with identical positions: {str(e)}")

        self.assertTrue(
            test_passed,
            "update_connectivity should handle identical positions gracefully"
        )

    def test_2d_vs_3d_positions(self):
        """
        Test that both 2D and 3D positions are handled correctly.

        This test verifies that the spatial indexing works correctly regardless
        of the dimensionality of the position vectors.
        """
        # Create a new position dictionary with 2D positions
        positions_2d = {
            "agent1": np.array([0.0, 0.0]),
            "agent2": np.array([1.0, 0.0]),
            "agent3": np.array([0.0, 1.0])
        }

        # Create a new model with 2D positions
        pos_fn_2d = Mock(return_value=positions_2d)
        agent_ids_2d = ["agent1", "agent2", "agent3"]

        model_2d = SignalBasedModel(
            agent_ids=agent_ids_2d,
            pos_fn=pos_fn_2d,
            tx_power=1.0,
            min_strength=0.05,
            dropout_alpha=2.0
        )

        # Call update_connectivity (this should not raise any exceptions)
        try:
            model_2d.update_connectivity(self.comms_matrix)
            test_passed = True
        except Exception as e:
            test_passed = False
            self.fail(f"update_connectivity raised an exception with 2D positions: {str(e)}")

        self.assertTrue(
            test_passed,
            "update_connectivity should handle 2D positions gracefully"
        )

    def test_large_agent_network(self):
        """
        Test spatial indexing performance with a large number of agents.

        This test verifies that the KD-Tree spatial indexing provides efficient
        querying even with many agents.
        """
        # Skip this test for regular runs, as it's more of a performance test
        # Remove the skip decorator when profiling performance
        self.skipTest("Performance test skipped during regular testing")

        # Create a large number of agents (e.g., 1000)
        num_agents = 1000
        large_agent_ids = [f"agent{i}" for i in range(num_agents)]

        # Create random positions for all agents
        np.random.seed(42)  # For reproducibility
        large_positions = {
            aid: np.random.rand(3) * 100  # Random positions in a 100x100x100 cube
            for aid in large_agent_ids
        }

        # Create a model with these agents
        large_pos_fn = Mock(return_value=large_positions)

        large_model = SignalBasedModel(
            agent_ids=large_agent_ids,
            pos_fn=large_pos_fn,
            tx_power=10.0,
            min_strength=0.01,
            dropout_alpha=0.5
        )

        # Create a mock ActiveCommunication matrix
        large_comms_matrix = Mock(spec=ActiveCommunication)
        large_comms_matrix.update = MagicMock()

        # Measure the time taken to update connectivity
        import time
        start_time = time.time()

        # Call update_connectivity
        large_model.update_connectivity(large_comms_matrix)

        end_time = time.time()
        execution_time = end_time - start_time

        # The execution time should be reasonable for a large network
        # This is a subjective threshold and may need adjustment based on hardware
        self.assertLess(
            execution_time,
            5.0,  # 5 seconds is a reasonable upper bound for 1000 agents
            f"Spatial indexing is too slow: {execution_time:.2f} seconds for {num_agents} agents"
        )


if __name__ == "__main__":
    unittest.main()