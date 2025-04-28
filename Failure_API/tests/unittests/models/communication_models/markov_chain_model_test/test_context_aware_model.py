import pytest
import numpy as np

from Failure_API.src.failure_api.communication_models.context_aware_markov_model import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.scenarios.markov.modifiers import create_weather_modifier

class TestContextAwareMarkovModel:
    """Test the that context-aware Markov model initializes correctly with the modifiers"""
    def test_initialization(self, agent_ids, transition_probabilities):

        """Test if the model initializes correctly"""
        def test_modifier(matrix, sender, receiver):
            return matrix.copy()

        modifiers = [test_modifier]
        model = ContextAwareMarkovModel(agent_ids, transition_probabilities, modifiers)
        model.rng = np.random.RandomState(seed=42)

        # check that the model stores the modifiers correctly
        assert len(model.modifier_functions) == 1
        assert model.modifier_functions[0] is test_modifier

    def test_matrix_modification_behavior(self, context_aware_model, transition_probabilities):
        """ Test that the model applies the modifiers to transition matrices."""

        # add test modifiers to the model
        def add_constant(matrix, sender, receiver):
            added = matrix.copy() + 0.1
            return added

        context_aware_model.modifier_functions = [add_constant]

        # Get modified matrix
        matrix = context_aware_model.get_transition_matrix("agent_0", "agent_1")

        # The original matrix plus 0.1, then normalized to keep probabilities valid
        original = transition_probabilities[("agent_0", "agent_1")]
        expected_row0 = (original[0] + 0.1) / (original[0] + 0.1).sum()
        expected_row1 = (original[1] + 0.1) / (original[1] + 0.1).sum()

        # Check with tolerance for floating point errors
        np.testing.assert_allclose(matrix[0], expected_row0, rtol=1e-5)
        np.testing.assert_allclose(matrix[1], expected_row1, rtol=1e-5)

    def test_modifier_sequence_application(self, context_aware_model, transition_probabilities):
        """Test that multiple modifiers are applied in the correct sequence."""

        # Define modifiers that can be tracked
        def first_modifier(matrix, sender, receiver):
            # Add 0.1 to the first element of each row
            result = matrix.copy()
            result[0, 0] += 0.1
            result[1, 0] += 0.1
            return result

        def second_modifier(matrix, sender, receiver):
            # Multiply the second element of each row by 2
            result = matrix.copy()
            result[0, 1] *= 2
            result[1, 1] *= 2
            return result

        # Add the modifiers
        context_aware_model.modifier_functions = [first_modifier, second_modifier]

        # Get the modified matrix
        matrix = context_aware_model.get_transition_matrix("agent_0", "agent_1")

        # The original matrix with first_modifier then second_modifier applied
        original = transition_probabilities[("agent_0", "agent_1")].copy()

        # Apply first modifier
        temp = original.copy()
        temp[0, 0] += 0.1
        temp[1, 0] += 0.1

        # Apply the second modifier
        temp[0, 1] *= 2
        temp[1, 1] *= 2

        # Normalize
        expected_row0 = temp[0] / temp[0].sum()
        expected_row1 = temp[1] / temp[1].sum()

        # Check results
        np.testing.assert_allclose(matrix[0], expected_row0, rtol=1e-5)
        np.testing.assert_allclose(matrix[1], expected_row1, rtol=1e-5)

    def test_behavioral_outcome(self, context_aware_model, active_communication, mock_weather_function):
        """Test the actual behavioral outcome of the context-aware model."""

        context_aware_model.rng = np.random.RandomState(seed=42)

        # Set up a model with the weather modifier
        context_aware_model.modifier_functions = [
            create_weather_modifier(mock_weather_function)
        ]

        # Update connectivity
        context_aware_model.update_connectivity(active_communication)

        # Check behavior against expected outcomes based on seed
        assert active_communication.can_communicate("agent_0", "agent_1") is True
        assert active_communication.can_communicate("agent_1", "agent_2") is True



