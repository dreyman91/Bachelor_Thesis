import pytest
import numpy as np
from Failure_API.src.failure_api.scenarios.markov.modifiers import create_time_modifier, create_traffic_modifier, create_weather_modifier

class TestModifiers:

    def test_weather_modifier(self, mock_weather_function):
        """Test the weather modifier"""
        w_modifier = create_weather_modifier(mock_weather_function)
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])

        # apply modifier
        modified = w_modifier(test_matrix, "agent_0", "agent_1")

        # Only the bottom row should change
        np.testing.assert_array_equal(modified[0], test_matrix[0])

        # Expect increased disconnection probability
        expected = np.array([0.8 + 0.1, 0.2 - 0.1])
        np.testing.assert_array_equal(modified[1], expected)

    def test_time_modifier(self, mock_time_function):
        """Test that  the time modifier correctly adjusts probabilities based on time."""

        # create the time modifier with a mock function
        time_modifier = create_time_modifier(mock_time_function)

        # create a test matrix
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])

        modified = time_modifier(test_matrix, "agent_0", "agent_1")

        np.testing.assert_array_equal(modified[0], test_matrix[0])

        expected = np.array([0.8 + 0.05, 0.2 - 0.05])
        np.testing.assert_array_equal(modified[1], expected)

    def test_time_modifier_no_effect(self):
        """Test the time modifier when it shouldn't have an effect."""

        # Create a function that returns a time not divisible by 10
        def time_not_divisible():
            return 11

        time_modifier = create_time_modifier(time_not_divisible)
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])
        modified = time_modifier(test_matrix, "agent1", "agent2")

        # Matrix should remain unchanged
        np.testing.assert_array_equal(modified, test_matrix)

    def test_traffic_modifier(self, mock_traffic_function):
        """Test that the traffic modifier correctly adjusts probabilities based on traffic."""

        # Create the traffic modifier
        traffic_modifier = create_traffic_modifier(mock_traffic_function)

        # Create a test matrix
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])

        # Apply the modifier
        modified = traffic_modifier(test_matrix, "agent1", "agent2")

        # Only the bottom row should change
        np.testing.assert_array_equal(modified[0], test_matrix[0])

        # Expect increased disconnection probability
        expected = np.array([0.8 + 0.3, 0.2 - 0.3])
        np.testing.assert_array_equal(modified[1], expected)

    def test_traffic_modifier_different_link(self, mock_traffic_function):
        """Test that the traffic modifier doesn't affect unrelated links."""

        # Create the traffic modifier
        traffic_modifier = create_traffic_modifier(mock_traffic_function)

        # Create a test matrix
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])

        # Apply the modifier to a different link
        modified = traffic_modifier(test_matrix, "agent2", "agent3")

        # The matrix should remain unchanged
        np.testing.assert_array_equal(modified, test_matrix)

    def test_modifier_composition(self, mock_weather_function, mock_traffic_function):
        """Test that modifiers can be composed correctly."""

        # Create the modifiers
        weather_modifier = create_weather_modifier(mock_weather_function)
        traffic_modifier = create_traffic_modifier(mock_traffic_function)

        # Create a test matrix
        test_matrix = np.array([[0.9, 0.1], [0.8, 0.2]])

        # Apply modifiers in a sequence
        after_weather = weather_modifier(test_matrix, "agent1", "agent2")
        after_both = traffic_modifier(after_weather, "agent1", "agent2")

        # Only the bottom row should change
        np.testing.assert_array_equal(after_both[0], test_matrix[0])

        # Expect cumulative effect
        expected = np.array([0.8 + 0.1 + 0.3, 0.2 - 0.1 - 0.3])
        np.testing.assert_array_equal(after_both[1], expected)