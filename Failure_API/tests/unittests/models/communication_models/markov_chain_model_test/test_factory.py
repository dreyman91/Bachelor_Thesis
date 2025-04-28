import pytest
import numpy as np
from Failure_API.src.failure_api.scenarios.markov.factory import create_context_aware_markov_model
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

def test_factory_creates_model_with_correct_structure(agent_ids, transition_probabilities,
                                                      mock_weather_function, mock_time_function,
                                                      mock_traffic_function):
    """Test that the factory correctly creates a model with the appropriate modifiers."""

    # Create a model with all modifiers
    model = create_context_aware_markov_model(
        agent_ids,
        transition_probabilities,
        context_fn=mock_weather_function,
        time_fn=mock_time_function,
        traffic_fn=mock_traffic_function
    )

    # verify the model structure
    assert model.agent_ids == agent_ids
    assert model.transition_probabilities == transition_probabilities
    assert len(model.modifier_functions) == 3

    # Make it deterministic
    model.rng = np.random.RandomState(99)

    return model

def test_factory_create_model_with_expected_behavior(agent_ids, transition_probabilities,
                                                     mock_weather_function, mock_time_function,
                                                     mock_traffic_function):
    """Test that the factory correctly creates a model that behaves as expected."""

    # create a model with all modifiers
    model = create_context_aware_markov_model(
        agent_ids,
        transition_probabilities,
        context_fn=mock_weather_function,
        time_fn=mock_time_function,
        traffic_fn=mock_traffic_function
    )

    model.rng = np.random.RandomState(99)

    # create a communication matrix and update it
    comms_matrix = ActiveCommunication(agent_ids)
    model.update_connectivity(comms_matrix)

    assert comms_matrix.can_communicate("agent_0", "agent_1") is True
    assert comms_matrix.can_communicate("agent_1", "agent_2") is False
    assert comms_matrix.can_communicate("agent_0", "agent_2") is True

def test_factory_with_partial_modifiers(agent_ids, transition_probabilities, mock_weather_function):
    """Test that the factory correctly handles partial modifier configurations."""

    model = create_context_aware_markov_model(
        agent_ids,
        transition_probabilities
    )

    # Verify the model structure
    assert model.agent_ids == agent_ids
    assert model.transition_probabilities == transition_probabilities
    assert len(model.modifier_functions) == 0


