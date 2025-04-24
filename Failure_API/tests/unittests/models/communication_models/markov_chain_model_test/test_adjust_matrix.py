"""
Test module for verifying the matrix adjustment functionality of the ContextAwareMarkovModel.

This module tests how the model adjusts transition probability matrices based on
context, time, and traffic conditions.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from Failure_API.src.failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestMatrixAdjustment(unittest.TestCase):
    """Test suite for transition matrix adjustment in ContextAwareMarkovModel."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_ids = ["agent1", "agent2", "agent3"]
        self.base_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.sender = "agent1"
        self.receiver = "agent2"

        # Empty transition probabilities - we'll be testing the adjustment directly
        self.transition_probabilities = {}

    def test_no_adjustment(self):
        """Test that matrix isn't modified when no context functions are provided."""
        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities
        )

        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)

        # The matrix should be a copy, not the same object
        self.assertIsNot(adjusted, self.base_matrix)

        # But values should be identical
        np.testing.assert_array_equal(adjusted, self.base_matrix)

    def test_weather_adjustment(self):
        """Test matrix adjustment based on weather context."""
        # Mock context function that reports bad weather
        context_fn = lambda: {"weather": 0.6}  # > 0.5 threshold

        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            context_fn=context_fn
        )

        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)

        # Check adjustment for bad weather - should increase reconnection probability
        expected = np.array([[0.9, 0.1], [0.3, 0.7]])  # [1][0] += 0.1, [1][1] -= 0.1
        np.testing.assert_array_almost_equal(adjusted, expected)

        # Verify row sums are still 1.0 (valid probability distribution)
        self.assertAlmostEqual(adjusted[0].sum(), 1.0)
        self.assertAlmostEqual(adjusted[1].sum(), 1.0)

    def test_time_adjustment(self):
        """Test matrix adjustment based on time conditions."""
        # Mock time function that returns a time divisible by 10
        time_fn = lambda: 20  # Should trigger adjustment

        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            time_fn=time_fn
        )

        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)

        # Check adjustment for special time - should increase reconnection probability
        expected = np.array([[0.9, 0.1], [0.25, 0.75]])  # [1][0] += 0.05, [1][1] -= 0.05
        np.testing.assert_array_almost_equal(adjusted, expected)

        # Verify row sums are still 1.0
        self.assertAlmostEqual(adjusted[0].sum(), 1.0)
        self.assertAlmostEqual(adjusted[1].sum(), 1.0)

        # Test with non-triggering time
        model.time_fn = lambda: 21  # Not divisible by 10
        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)
        np.testing.assert_array_equal(adjusted, self.base_matrix)

    def test_traffic_adjustment(self):
        """Test matrix adjustment based on traffic conditions."""
        # Mock traffic function that reports high traffic
        traffic_fn = lambda: {("agent1", "agent2"): 0.6}  # > 0.5 threshold

        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            traffic_fn=traffic_fn
        )

        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)

        # Check adjustment for high traffic - should increase reconnection probability
        expected = np.array([[0.9, 0.1], [0.5, 0.5]])  # [1][0] += 0.1, [1][1] -= 0.1
        np.testing.assert_array_almost_equal(adjusted, expected)

        # Verify row sums are still 1.0
        self.assertAlmostEqual(adjusted[0].sum(), 1.0)
        self.assertAlmostEqual(adjusted[1].sum(), 1.0)

        # Test with different sender-receiver pair not in traffic dict
        adjusted = model._adjust_matrix(self.base_matrix, "agent2", "agent3")
        np.testing.assert_array_equal(adjusted, self.base_matrix)

    def test_combined_adjustments(self):
        """Test matrix adjustment with all context factors combined."""
        context_fn = lambda: {"weather": 0.6}  # Trigger weather adjustment
        time_fn = lambda: 20  # Trigger time adjustment
        traffic_fn = lambda: {("agent1", "agent2"): 0.6}  # Trigger traffic adjustment

        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            context_fn=context_fn,
            time_fn=time_fn,
            traffic_fn=traffic_fn
        )

        adjusted = model._adjust_matrix(self.base_matrix, self.sender, self.receiver)

        # Check combined adjustments:

        expected = np.array([[0.9, 0.1], [0.65, 0.35]])
        np.testing.assert_array_almost_equal(adjusted, expected)

        # Verify row sums are still 1.0
        self.assertAlmostEqual(adjusted[0].sum(), 1.0)
        self.assertAlmostEqual(adjusted[1].sum(), 1.0)

    def test_matrix_clamping(self):
        """Test that adjusted probabilities are clamped between 0 and 1."""
        # Create a matrix that will exceed bounds after adjustments
        extreme_matrix = np.array([[0.9, 0.1], [0.95, 0.05]])

        # Set up all adjustment functions to maximize the effect
        context_fn = lambda: {"weather": 1.0}
        time_fn = lambda: 10
        traffic_fn = lambda: {("agent1", "agent2"): 1.0}

        model = ContextAwareMarkovModel(
            agent_ids=self.agent_ids,
            transition_probabilities=self.transition_probabilities,
            context_fn=context_fn,
            time_fn=time_fn,
            traffic_fn=traffic_fn
        )

        adjusted = model._adjust_matrix(extreme_matrix, self.sender, self.receiver)

        # The adjustments would push [1][0] to 1.2, which should be clamped to 1.0
        # And [1][1] to -0.2, which should be clamped to 0.0
        # After normalization to ensure rows sum to 1, we should get:
        expected = np.array([[0.9, 0.1], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(adjusted, expected)

        # Verify row sums are still 1.0
        self.assertAlmostEqual(adjusted[0].sum(), 1.0)
        self.assertAlmostEqual(adjusted[1].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()