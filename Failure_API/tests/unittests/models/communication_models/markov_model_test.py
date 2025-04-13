import unittest
import numpy as np
from Failure_API.src.failure_api.communication_models.markov_chain_based import ContextAwareMarkovModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

class TestMarkovModel(unittest.TestCase):
    def setUp(self):
        self.agents = ["a0", "a1"]
        self.initial_matrix = np.ones((2, 2), dtype=bool)
        np.fill_diagonal(self.initial_matrix, False)

    def test_basic_markov_update(self):
        """Check if the model updates the communication state between agents"""
        t_p = {
            ("a0", "a1"): np.array([[0.9, 0.1],
                                    [0.1, 0.9]]),
            ("a1", "a0"): np.array([[0.8, 0.2],
                                    [0.3, 0.7]])
        }

        model = ContextAwareMarkovModel(agent_ids=self.agents, transition_probabilities=t_p)
        ac = ActiveCommunication(self.agents)

        model.set_rng(np.random.default_rng(42))
        model.update_connectivity(ac)

        matrix = ac.get_state()
        self.assertIn(matrix[0, 1], [True, False])
        self.assertIn(matrix[1, 0], [True, False])

    def test_context_effect_on_transition(self):

        t_p = {
            ("a0", "a1"): np.array([[0.9, 0.1], [0.1, 0.9]])
        }
        def context_fn():
            return {"weather": 1.0}

        model = ContextAwareMarkovModel(agent_ids=self.agents, transition_probabilities=t_p,
                                        context_fn=context_fn)
        model.set_rng(np.random.default_rng(1))
        ac = ActiveCommunication(self.agents)
        model.update_connectivity(ac)

        state = model.state[("a0", "a1")]
        self.assertIn(state, [0, 1])
