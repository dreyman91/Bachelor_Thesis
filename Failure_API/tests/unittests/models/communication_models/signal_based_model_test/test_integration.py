
"""
Integration Tests for SignalBasedModel

This test file focuses on testing the integration of all components of the
SignalBasedModel class, verifying that they work together correctly in
realistic scenarios including:

1. Full signal propagation pipeline from position to connectivity
2. Integration with actual ActiveCommunication instances
3. Behavior with dynamic agent movements
4. Compatibility with the PettingZoo framework
5. Composability with other communication models
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, List, Callable
import math

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class RealActiveCommunication:
    """
    A simplified implementation of ActiveCommunication for testing.

    This class provides a more realistic implementation of the
    ActiveCommunication interface for integration testing.
    """

    def __init__(self, agent_ids):
        """Initialize with a set of agent IDs."""
        self.agent_ids = agent_ids
        self.id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
        # Initialize matrix with all 1s (all connected)
        self.matrix = np.ones((len(agent_ids), len(agent_ids)), dtype=bool)
        # Set diagonal to False (no self-connections)
        np.fill_diagonal(self.matrix, False)

    def update(self, sender: str, receiver: str, connected: bool):
        """Update the connectivity matrix."""
        sender_idx = self.id_to_idx[sender]
        receiver_idx = self.id_to_idx[receiver]
        self.matrix[sender_idx, receiver_idx] = connected

    def get(self, sender: str, receiver: str) -> bool:
        """Get the connectivity status between two agents."""
        sender_idx = self.id_to_idx[sender]
        receiver_idx = self.id_to_idx[receiver]
        return self.matrix[sender_idx, receiver_idx]

    def get_connectivity_matrix(self) -> np.ndarray:
        """Get the full connectivity matrix."""
        return self.matrix.copy()


class DynamicAgentEnvironment:
    """
    A simulated environment with moving agents for testing.

    This class provides a way to test the SignalBasedModel with
    dynamically changing agent positions over time.
    """

    def __init__(self, agent_ids: List[str], arena_size: float = 10.0):
        """
        Initialize the environment with fixed agent IDs and arena size.

        Args:
            agent_ids: List of unique agent identifiers
            arena_size: Size of the square arena (agents move within this space)
        """
        self.agent_ids = agent_ids
        self.arena_size = arena_size

        # Initialize random positions
        np.random.seed(42)  # For reproducibility
        self.positions = {
            aid: np.random.rand(2) * arena_size
            for aid in agent_ids
        }

        # Initialize random velocities (change per step)
        self.velocities = {
            aid: (np.random.rand(2) - 0.5) * 1.0  # Random direction, Quick movement
            for aid in agent_ids
        }

    def get_positions(self) -> Dict[str, np.ndarray]:
        """Get the current positions of all agents."""
        return {aid: pos.copy() for aid, pos in self.positions.items()}

    def step(self):
        """
        Update agent positions based on their velocities.

        Implements simple bouncing behavior at arena boundaries.
        """
        for aid in self.agent_ids:
            # Update position
            self.positions[aid] += self.velocities[aid]

            # Bounce if hitting boundaries
            for dim in range(2):
                if self.positions[aid][dim] < 0:
                    self.positions[aid][dim] = -self.positions[aid][dim]
                    self.velocities[aid][dim] = -self.velocities[aid][dim]
                elif self.positions[aid][dim] > self.arena_size:
                    self.positions[aid][dim] = 2 * self.arena_size - self.positions[aid][dim]
                    self.velocities[aid][dim] = -self.velocities[aid][dim]


class TestSignalBasedModelIntegration(unittest.TestCase):
    """
    Test class for verifying the integrated functionality of SignalBasedModel.

    These tests ensure that all components of the model work together correctly
    in realistic scenarios and are compatible with expected usage patterns.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        Creates a realistic environment with multiple agents for integration testing.
        """
        # Create a list of agent IDs
        self.agent_ids = [f"agent{i}" for i in range(5)]

        # Create a dynamic environment for testing
        self.environment = DynamicAgentEnvironment(self.agent_ids, arena_size=10.0)

        # Create a model with default parameters
        self.model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.environment.get_positions,
            tx_power=5.0,  # Medium transmission power
            min_strength=0.05,  # Standard minimum strength
            dropout_alpha=1.0  # Moderate dropout with distance
        )

        # Create an ActiveCommunication instance
        self.comms_matrix = RealActiveCommunication(self.agent_ids)

    def test_full_update_pipeline(self):
        """
        Test the full signal propagation pipeline from positions to connectivity.

        This test verifies that the complete pipeline works end-to-end:
        1. Getting positions from the environment
        2. Building the KD-Tree for spatial indexing
        3. Computing signal strengths based on distances
        4. Determining connectivity based on signal thresholds
        5. Applying probabilistic packet loss
        6. Updating the connectivity matrix
        """
        # Fix random seed for reproducibility
        np.random.seed(42)

        # Call update_connectivity
        self.model.update_connectivity(self.comms_matrix)

        # Verify that the connectivity matrix has been updated
        # We can't predict exact values due to probabilistic factors,
        # but we can verify some properties
        connectivity_matrix = self.comms_matrix.get_connectivity_matrix()

        # Diagonal should be False (no self-connections)
        for i in range(len(self.agent_ids)):
            self.assertFalse(
                connectivity_matrix[i, i],
                "Self-connections should be False"
            )

        # There should be a mix of True and False values (not all connected)
        self.assertTrue(
            np.any(connectivity_matrix) and not np.all(connectivity_matrix),
            "Connectivity matrix should have a mix of True and False values"
        )

        # Connections should be symmetric for fixed values (ignoring probability)
        # We can't easily test this with probabilistic factors

    def test_moving_agents(self):
        """
        Test the model's behavior with dynamically moving agents.

        This test verifies that the connectivity changes appropriately as
        agents move around the environment.
        """
        # Run a simulation for several steps
        num_steps = 10
        connectivity_history = []

        for _ in range(num_steps):
            # Update agent positions
            self.environment.step()

            # Update connectivity
            self.model.update_connectivity(self.comms_matrix)

            # Record connectivity matrix
            connectivity_history.append(
                self.comms_matrix.get_connectivity_matrix().copy()
            )

        # Verify that connectivity changes over time
        # Compare first and last connectivity matrices
        first_matrix = connectivity_history[0]
        last_matrix = connectivity_history[-1]

        # They should be different due to agent movement
        self.assertFalse(
            np.array_equal(first_matrix, last_matrix),
            "Connectivity should change as agents move"
        )

        # Calculate the average number of connections over time
        connection_counts = [np.sum(matrix) for matrix in connectivity_history]
        avg_connections = sum(connection_counts) / len(connection_counts)

        # Should be a reasonable number (not all or none)
        max_possible = len(self.agent_ids) * (len(self.agent_ids) - 1)
        connection_ratio = avg_connections / max_possible

        self.assertTrue(
            0.01 < connection_ratio < 0.99,
            f"Connection ratio ({connection_ratio:.2f}) should be reasonable"
        )

    def test_pettingzoo_compatibility(self):
        """
        Test compatibility with the PettingZoo framework.

        This test verifies that the model's interface and behavior are
        compatible with expected PettingZoo usage patterns.
        """
        # In PettingZoo, the typical usage pattern would be:
        # 1. Initialize the model with a list of agent IDs and position function
        # 2. Create an initial connectivity matrix
        # 3. Update the connectivity matrix in each step of the environment

        # Create initial connectivity matrix
        initial_matrix = SignalBasedModel.create_initial_matrix(self.agent_ids)

        # Verify that all off-diagonal values are True
        off_diagonal = initial_matrix[~np.eye(len(self.agent_ids), dtype=bool)]
        self.assertTrue(
            np.all(off_diagonal),
            "All off-diagonal values should be True"
        )

        print("Initial matrix:\n", initial_matrix)
        print("Off-diagonal values:\n", off_diagonal)
        print("Unique values in off-diagonal:", np.unique(off_diagonal))
        # Simulate environment steps
        for _ in range(5):
            # Update environment
            self.environment.step()

            # Update connectivity
            self.model.update_connectivity(self.comms_matrix)

        # Verify that the model can be reset or reinitialized
        new_agent_ids = self.agent_ids + ["new_agent"]
        new_environment = DynamicAgentEnvironment(new_agent_ids)

        new_model = SignalBasedModel(
            agent_ids=new_agent_ids,
            pos_fn=new_environment.get_positions
        )

        # Create a new communication matrix
        new_comms_matrix = RealActiveCommunication(new_agent_ids)

        # Update connectivity with the new model
        new_model.update_connectivity(new_comms_matrix)

        # Verify the new matrix shape
        connectivity_matrix = new_comms_matrix.get_connectivity_matrix()
        self.assertEqual(
            connectivity_matrix.shape,
            (len(new_agent_ids), len(new_agent_ids)),
            "New connectivity matrix should have updated shape"
        )

    def test_model_composition(self):
        """
        Test composition with other communication models.

        This test verifies that the SignalBasedModel can be combined with
        other communication models (in sequence or in parallel).
        """

        # Create a simple mock model that flips all connections
        class FlipperModel(CommunicationModels):
            def __init__(self, agent_ids):
                super().__init__()
                self.agent_ids = agent_ids

            def update_connectivity(self, comms_matrix):
                # Flip every connection
                for i, sender in enumerate(self.agent_ids):
                    for j, receiver in enumerate(self.agent_ids):
                        if sender != receiver:
                            current = comms_matrix.get(sender, receiver)
                            comms_matrix.update(sender, receiver, not current)

            @staticmethod
            def create_initial_matrix(agent_ids: List[str]) -> np.ndarray:
                n = len(agent_ids)
                matrix = np.ones((n, n))
                np.fill_diagonal(matrix, 0.0)
                return matrix

        # First, update with SignalBasedModel
        self.model.update_connectivity(self.comms_matrix)

        # Record the state after SignalBasedModel update
        signal_based_state = self.comms_matrix.get_connectivity_matrix().copy()

        # Create and apply the flipper model
        flipper_model = FlipperModel(self.agent_ids)
        flipper_model.update_connectivity(self.comms_matrix)

        # Record the state after FlipperModel update
        flipped_state = self.comms_matrix.get_connectivity_matrix().copy()

        # All connections should be flipped (except diagonal)
        for i in range(len(self.agent_ids)):
            for j in range(len(self.agent_ids)):
                if i != j:  # Skip diagonal
                    self.assertEqual(
                        signal_based_state[i, j],
                        not flipped_state[i, j],
                        f"Connection ({i},{j}) should be flipped"
                    )


    def test_parameterized_behavior(self):
        """
        Test that model parameters have the expected effects on behavior.

        This test verifies that changing the model parameters (tx_power,
        min_strength, dropout_alpha) has the expected effects on the
        resulting connectivity.
        """
        # Define a set of parameter configurations to test
        configurations = [
            # High power, low threshold, low dropout -> Many connections
            {
                "tx_power": 10.0,
                "min_strength": 0.01,
                "dropout_alpha": 0.1,
                "expected_connectivity": "high"
            },
            # Low power, high threshold, high dropout -> Few connections
            {
                "tx_power": 1.0,
                "min_strength": 0.2,
                "dropout_alpha": 5.0,
                "expected_connectivity": "low"
            }
        ]

        # Fix environment for consistent testing
        fixed_environment = DynamicAgentEnvironment(self.agent_ids)

        # Test each configuration
        connectivity_ratios = []

        for config in configurations:
            # Create a model with the config parameters
            model = SignalBasedModel(
                agent_ids=self.agent_ids,
                pos_fn=fixed_environment.get_positions,
                tx_power=config["tx_power"],
                min_strength=config["min_strength"],
                dropout_alpha=config["dropout_alpha"]
            )

            # Create a fresh communication matrix
            comms_matrix = RealActiveCommunication(self.agent_ids)

            # Fix random seed for reproducibility
            np.random.seed(42)

            # Update connectivity
            model.update_connectivity(comms_matrix)

            # Calculate connection ratio
            connectivity = comms_matrix.get_connectivity_matrix()
            connection_count = np.sum(connectivity)
            max_possible = len(self.agent_ids) * (len(self.agent_ids) - 1)
            ratio = connection_count / max_possible

            connectivity_ratios.append(ratio)

        # Verify that high expected connectivity has higher ratio than low
        self.assertGreater(
            connectivity_ratios[0],  # High expected connectivity
            connectivity_ratios[1],  # Low expected connectivity
            "Parameters should have the expected effect on connectivity"
        )

    def test_large_scale_integration(self):
        """
        Test integration with a larger number of agents.

        This test verifies that the model scales effectively to handle
        a larger number of agents, as might be found in real applications.
        """
        # Skip this test for regular runs, as it's more of a performance test
        # Remove the skip decorator when testing scalability
        self.skipTest("Performance test skipped during regular testing")

        # Create a large number of agents
        num_agents = 100
        large_agent_ids = [f"agent{i}" for i in range(num_agents)]

        # Create a large environment
        large_environment = DynamicAgentEnvironment(large_agent_ids, arena_size=50.0)

        # Create a model for the large environment
        large_model = SignalBasedModel(
            agent_ids=large_agent_ids,
            pos_fn=large_environment.get_positions,
            tx_power=10.0,
            min_strength=0.01,
            dropout_alpha=0.5
        )

        # Create a communication matrix
        large_comms_matrix = RealActiveCommunication(large_agent_ids)

        # Measure update time
        import time
        start_time = time.time()

        # Update connectivity
        large_model.update_connectivity(large_comms_matrix)

        end_time = time.time()
        update_time = end_time - start_time

        # The update should complete in a reasonable time
        self.assertLess(
            update_time,
            1.0,  # 1 second is a reasonable upper bound for 100 agents
            f"Large-scale update took too long: {update_time:.2f} seconds"
        )

        # Verify some basic properties of the resulting connectivity
        connectivity = large_comms_matrix.get_connectivity_matrix()

        # Diagonal should be False
        for i in range(num_agents):
            self.assertFalse(connectivity[i, i], "Self-connections should be False")

        # There should be a mix of True and False values
        self.assertTrue(
            np.any(connectivity) and not np.all(connectivity),
            "Connectivity matrix should have a mix of True and False values"
        )


if __name__ == "__main__":
    unittest.main()