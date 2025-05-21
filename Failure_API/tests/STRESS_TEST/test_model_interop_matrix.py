# test_model_interop_matrix.py

import pytest
import numpy as np
from Failure_API.src.failure_api.communication_models import (
    DelayBasedModel, DistanceModel, SignalBasedModel, BaseMarkovModel
)
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

AGENTS = ["agent_0", "agent_1", "agent_2"]

def dummy_pos(agent=None):
    pos = {
        "agent_0": np.array([0, 0]),
        "agent_1": np.array([1, 0]),
        "agent_2": np.array([10, 10])
    }
    return pos if agent is None else pos[agent]

def build_matrix():
    return ActiveCommunication(AGENTS)

def test_combined_markov_and_signal_model():
    matrix = build_matrix()
    m1 = BaseMarkovModel(AGENTS, default_matrix=np.array([[0.8, 0.2], [0.3, 0.7]]))
    m2 = SignalBasedModel(AGENTS, pos_fn=dummy_pos)
    m1.update_connectivity(matrix)
    m2.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert isinstance(mat[i, j], (bool, np.bool_))

def test_delay_and_distance_interaction():
    matrix = build_matrix()
    d1 = DelayBasedModel(AGENTS, min_delay=0, max_delay=1)
    d2 = DistanceModel(AGENTS, distance_threshold=5.0, pos_fn=dummy_pos)
    d1.update_connectivity(matrix)
    d2.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)

def test_asymmetric_links():
    matrix = ActiveCommunication(["a", "b"])
    model = DelayBasedModel(["a", "b"], min_delay=0, max_delay=1, )
    model.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat[0, 1] != mat[1, 0]

def test_all_models_combined():
    agents = ["a", "b", "c"]
    matrix = ActiveCommunication(agents)

    models = [
        DistanceModel(agents, distance_threshold=5.0, pos_fn=lambda a=None: {
            "a": np.array([0, 0]), "b": np.array([1, 0]), "c": np.array([10, 10])
        } if a is None else {"a": np.array([0, 0]), "b": np.array([1, 0]), "c": np.array([10, 10])}[a]),
        DelayBasedModel(agents, min_delay=0, max_delay=1),
        BaseMarkovModel(agents, default_matrix=np.array([[0.9, 0.1], [0.2, 0.8]])),
        SignalBasedModel(agents, pos_fn=lambda a=None: {
            "a": np.array([0, 0]), "b": np.array([1, 0]), "c": np.array([10, 10])
        } if a is None else {"a": np.array([0, 0]), "b": np.array([1, 0]), "c": np.array([10, 10])}[a])
    ]

    for model in models:
        model.update_connectivity(matrix)

    mat = matrix.get_boolean_matrix()
    assert mat.shape == (3, 3)
    for i in range(3):
        assert mat[i, i] == False  # self-links must be False


def test_stability_across_updates():
    agents = ["x", "y"]
    matrix = ActiveCommunication(agents)
    model = BaseMarkovModel(agents, default_matrix=np.array([[0.99, 0.01], [0.01, 0.99]]))

    for _ in range(50):
        model.update_connectivity(matrix)
        mat = matrix.get_boolean_matrix()
        for i in range(2):
            assert mat[i, i] == False
            for j in range(2):
                if i != j:
                    assert isinstance(mat[i, j], (bool, np.bool_))

def test_bandwidth_zero_threshold():
    agents = ["a", "b"]
    def distant_pos(a=None):
        return {"a": np.array([0, 0]), "b": np.array([999, 999])} if a is None else {"a": np.array([0, 0]), "b": np.array([999, 999])}[a]

    matrix = ActiveCommunication(agents)
    model = DistanceModel(agents, distance_threshold=1.0, pos_fn=distant_pos)
    model.update_connectivity(matrix)
    mat = matrix.get_boolean_matrix()
    assert mat[0, 1] == False
    assert mat[1, 0] == False









