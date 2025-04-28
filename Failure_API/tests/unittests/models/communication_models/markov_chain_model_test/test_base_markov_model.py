import pytest
import numpy as np

from Failure_API.src.failure_api.communication_models.base_markov_model import BaseMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.tests.unittests.models.communication_models.markov_chain_model_test.conftest import \
    active_communication
from Failure_API.tests.utils.test_common_utils import agent_ids

class TestBaseMarkovModel:

    def test_initialization(self, agent_ids, transition_probabilities, base_markov_model):
        """Test if the model initializes correctly"""

        # Check that the model stores agent IDs correctly
        assert base_markov_model.agent_ids == agent_ids

        # check that the model stores the transition probabilities correctly
        assert (base_markov_model.transition_probabilities[("agent_0", "agent_1")] ==
                transition_probabilities[("agent_0", "agent_1")]).all()

        # check that the initial state is "connected" i.e. (1)
        assert base_markov_model.state[("agent_0", "agent_1")] == 1

    def test_get_transition_matrix(self, base_markov_model, transition_probabilities):
        """Test that the model returns the correct transition matrix."""
        # Returns the specified matrix for known links
        matrix = base_markov_model.get_transition_matrix("agent_0", "agent_1")
        assert (matrix == transition_probabilities[("agent_0", "agent_1")]).all()

        # Returns default matrix for unknown links
        matrix = base_markov_model.get_transition_matrix("agent_0", "agent_2")
        assert matrix.shape == (2, 2)
        assert matrix[0, 0] == 0.9

    def test_update_pair(self, base_markov_model, active_communication, agent_ids):
        """Test that update_pair correctly updates link states"""
        base_markov_model._update_pair("agent_0", "agent_1", active_communication)

        # Update the pair and check the result in the connectivity matrix
        agent0_idx = agent_ids.index("agent_0")
        agent1_idx = agent_ids.index("agent_1")

        # Check actual behavior
        expected_connected = True
        actual_connected = active_communication.get_boolean_matrix()[agent1_idx, agent0_idx]
        assert actual_connected == expected_connected

    def test_update_connectivity(self, base_markov_model, active_communication):
        """Test the behavior of update_connectivity by validating the resulting connectivity matrix."""
        base_markov_model.update_connectivity(active_communication)

        # Check specific connections
        assert active_communication.can_communicate("agent_0", "agent_1") is True
        assert active_communication.can_communicate("agent_1", "agent_2") is True
        assert active_communication.can_communicate("agent_0", "agent_2") is True

    def test_create_initial_matrix(self, agent_ids):
        """ test that the static method returns the correct matrix"""

        matrix = BaseMarkovModel.create_initial_matrix(agent_ids)

        # check dimensions
        assert matrix.shape == (len(agent_ids), len(agent_ids))

        # check that diagonal elements are False (no sel-loops)
        for i in range(len(agent_ids)):
            assert not matrix[i, i]

        # check that not-diagonal elements are True
        for i in range(len(agent_ids)):
            for j in range(len(agent_ids)):
                if i != j:
                    assert matrix[i, j]

    def test_invalid_input_handling(self, agent_ids):
        """Test that the model handles invalid input correctly."""

        # Test with the wrong shape
        invalid_probs = {
            ("agent_0", "agent_1"): np.array([[1.5, 0.1]])
        }
        with pytest.raises(ValueError):
            BaseMarkovModel(agent_ids, invalid_probs)

        # Test with invalid transition probabilities
        invalid_probs = {
            ("agent_0", "agent_1"): np.array([[1.5, 0.1], [0.1, 0.9]]),
        }
        with pytest.raises(ValueError):
            BaseMarkovModel(agent_ids, invalid_probs)




