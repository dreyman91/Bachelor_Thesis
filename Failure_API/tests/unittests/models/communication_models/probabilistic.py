import unittest
from Failure_API.src.wrapper_api.connectivity_patterns.base_communication_model import ProbabilisticModel
from Failure_API.src.wrapper_api.connectivity_patterns.active_communication import ActiveCommunication
import numpy as np
from gymnasium import spaces

class TestProbabilisticModel(unittest.TestCase):

    def test_probabilistic_model(self):
        agents = ["agent0", "agent1", "agent2"]
        model = ProbabilisticModel(failure_prob=0.6, agent_ids=agents, seed=42)
        ac = ActiveCommunication(agent_ids=agents)

        model.update_connectivity(ac)
        matrix = ac.get_state()

        self.assertEqual(matrix.shape, (3,3))
        for i in range(3):
            self.assertTrue(matrix[i,i], "self communication must be True")

        print("Updated Probabilistic Matrix", matrix)





