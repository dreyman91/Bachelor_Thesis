import pytest
import numpy as np

from Failure_API.src.failure_api.communication_models.base_markov_model import BaseMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

AGENT_IDS = ["agent_0", "agent_1", "agent_2"]
TEST_TRANSITION_PROBS = {
    ("agent_0", "agent_1"): np.array([[0.9, 0.1], [0.1, 0.9]]),
    ("agent_1", "agent_2"): np.array([[0.8, 0.2], [0.3, 0.7]])
}

class TestBaseMarkovModel:

    def test_initialization(self):
        """Test if the model initializes correctly"""
        model = BaseMarkovModel(AGENT_IDS, TEST_TRANSITION_PROBS)

        # Check that the model stores agent IDs correctly
        assert model.agent_ids == AGENT_IDS

        # check that the model stores the transition probabilities correctly
        assert (model.transition_probabilities[("agent_0", "agent_1")] ==
                TEST_TRANSITION_PROBS[("agent_0", "agent_1")]).all()

        # check that the initial state is "connected" i.e. (1)
        assert model.state[("agent_0", "agent_1")] == 1

    def test_get_transition_matrix(self):
        """Test that the model returns the correct transition matrix."""
        model = BaseMarkovModel(AGENT_IDS, TEST_TRANSITION_PROBS)

        # Returns the specified matrix for known links
        matrix = model.get_transition_matrix("agent_0", "agent_1")
        assert (matrix == TEST_TRANSITION_PROBS[("agent_0", "agent_1")]).all()

        # Returns default matrix for unknown links
        matrix = model.get_transition_matrix("agent_0", "agent_3")
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0.9

    def test_update_pair(self):
        """Test that update_pair correctly updates link states"""
        model = BaseMarkovModel(AGENT_IDS, TEST_TRANSITION_PROBS)
        comms_matrix = ActiveCommunication(AGENT_IDS)

        # deterministic RNG for testing
        def mock_choice(options, p=None):
            return options[0] # Always disconnected for testing

        original_rng = model.rng
        model.rng = type('MockRNG', (), {'choice': mock_choice})

        # Update the pair
        model._update_pair("agent_0", "agent_1", comms_matrix)

        # Check the state was update correctly
        assert model.state[("agent_0", "agent_1")] == 0

        # check the communication matrix was updated correctly
        agent0_idx = AGENT_IDS.index("agent_0")
        agent1_idx = AGENT_IDS.index("agent_1")

        # check that the link is disconnected
        assert  not comms_matrix.get_boolean_matrix()[agent0_idx, agent1_idx]

        # restore the original RNG
        model.rng = original_rng

    def test_update_connectivity(self):
        """Test that update_connectivity correctly updates link states"""
        model = BaseMarkovModel(AGENT_IDS, TEST_TRANSITION_PROBS)
        comms_matrix = ActiveCommunication(AGENT_IDS)

        # deterministic RNG for testing
        def mock_choice(options, p=None):
            return options[0]

        original_rng = model.rng
        model.rng = type('MockRNG', (), {'choice': mock_choice})

        # update all links
        model.update_connectivity(comms_matrix)

        # check that all links are disconnected
        matrix = comms_matrix.get_boolean_matrix()
        for i in range(len(AGENT_IDS)):
            for j in range(len(AGENT_IDS)):
                if i != j:
                    assert not matrix[i, j]

        # restore original RNG
        model.rng = original_rng

    def test_create_initial_matrix(self):
        """ test that the static method returns the correct matrix"""
        matrix = BaseMarkovModel.create_initial_matrix(AGENT_IDS)

        # check dimensions
        assert matrix.shape == (len(AGENT_IDS), len(AGENT_IDS))

        # check that diagonal elements are False (no sel-loops)
        for i in range(len(AGENT_IDS)):
            assert not matrix[i, i]

        # check that not-diagonal elements are True
        for i in range(len(AGENT_IDS)):
            for j in range(len(AGENT_IDS)):
                if i != j:
                    assert matrix[i, j]



