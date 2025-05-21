# test_failure_assertions.py

import pytest
import numpy as np
from Failure_API.src.failure_api.communication_models import DelayBasedModel, BaseMarkovModel, DistanceModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

AGENTS = ["agent_0", "agent_1"]

def test_markov_model_invalid_matrix():
    with pytest.raises(ValueError):
        BaseMarkovModel(AGENTS, default_matrix=np.array([[1.0]]))

def test_delay_model_invalid_bounds():
    with pytest.raises(ValueError, match="min_delay cannot be greater than max_delay"):
        DelayBasedModel(AGENTS, min_delay=5, max_delay=2)

def test_distance_model_invalid_positions():
    def bad_fn(agent=None):
        return {
            "agent_0": np.array([0, 0]),
        }.get(agent, None)

    matrix = ActiveCommunication(AGENTS)
    model = DistanceModel(AGENTS, distance_threshold=1.0, pos_fn=bad_fn)
    with pytest.raises(ValueError, match="must be a numpy array"):
        model.update_connectivity(matrix)


def test_communication_matrix_shape_mismatch():
    with pytest.raises(ValueError, match="default_matrix must be 2x2"):
        BaseMarkovModel(["a", "b"], default_matrix=np.ones((3, 3)))

