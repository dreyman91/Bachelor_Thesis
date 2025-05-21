import numpy as np
import pytest
from Failure_API.src.failure_api.communication_models import (
    DelayBasedModel, DistanceModel, SignalBasedModel, BaseMarkovModel
)
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

AGENTS = ["agent_0", "agent_1", "agent_2"]


def dummy_pos_fn(agent=None):
    pos = {
        "agent_0": np.array([0.0, 0.0]),
        "agent_1": np.array([1.0, 0.0]),
        "agent_2": np.array([10.0, 10.0])  # far away
    }
    return pos if agent is None else pos.get(agent)


def create_comms():
    return ActiveCommunication(AGENTS)


def test_delay_model_behavior():
    model = DelayBasedModel(AGENTS, min_delay=1, max_delay=1, message_drop_probability=0.0)
    matrix = create_comms()
    model.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)
    for i in range(3):
        assert mat[i, i] == False


def test_distance_model_with_extreme_positions():
    model = DistanceModel(AGENTS, distance_threshold=5.0, pos_fn=dummy_pos_fn)
    matrix = create_comms()
    model.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)
    # agent_2 should not connect to others
    for i, agent in enumerate(AGENTS):
        if agent == "agent_2":
            assert not any(mat[i, :2])
            assert not any(mat[:2, i])


def test_signal_model_noise_and_range():
    model = SignalBasedModel(AGENTS, pos_fn=dummy_pos_fn, tx_power=10.0, min_strength=0.01, dropout_alpha=2.0)
    matrix = create_comms()
    model.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)
    # agent_2 should be isolated due to distance
    for i in range(2):
        assert mat[i, 2] == False


def test_markov_transition_matrix():
    default_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # locked states
    model = BaseMarkovModel(AGENTS, default_matrix=default_matrix)
    matrix = create_comms()
    model.update_connectivity(matrix)
    # All connections should remain unchanged
    for i, sender in enumerate(AGENTS):
        for j, receiver in enumerate(AGENTS):
            if sender != receiver:
                val = matrix.get(sender, receiver)
                assert val == True  # stays connected
