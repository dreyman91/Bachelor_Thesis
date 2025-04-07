import unittest
import numpy as np
from Failure_API.src.wrapper_api.models.communication_model import MatrixModel
from Failure_API.src.wrapper_api.models.active_communication import ActiveCommunication


class TestMatrixModel(unittest.TestCase):
    def test_matrix_model(self):
        agents = ["a0", "a1", "a2"]
        man_matrix = np.array([[1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]])
        fail_rate = 0.0
        model = MatrixModel(agent_ids=agents, comms_matrix_values=man_matrix, failure_prob=fail_rate, seed=None)
        ac = ActiveCommunication(agent_ids=agents)
        model.update_connectivity(ac)

        for i in range(3):
            for j in range(3):
                expected = man_matrix[i,j]
                actual = ac.get_state()[i,j]
                if fail_rate == 0.0:
                    self.assertTrue(actual == expected, "Matrix %s is equal to expected %s" % (i, j))
                else:
                    self.assertFalse(actual == expected, "Matrix is not equal to expected %s " % (i, j))