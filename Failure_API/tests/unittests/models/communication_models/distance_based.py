import unittest

from Failure_API.src.wrapper_api.connectivity_patterns.base_communication_model import DistanceModel
from Failure_API.src.wrapper_api.connectivity_patterns.active_communication import ActiveCommunication
import numpy as np


class TestDistanceBased(unittest.TestCase):
    def test_distance(self):
        agents = ["a0", "a1", "a2"]
    def pos_fn(self):
        return {
                "a0": np.array([0., 0.]),
                "a1": np.array([3., 0.]),
                "a2": np.array([6., 4.])
            }
        threshold = 3.0
        model1 = DistanceModel(agent_ids=agents, pos_fn=pos_fn, distance_threshold=threshold)
        ac = ActiveCommunication(agent_ids=agents)
        model1.update_connectivity(ac)

        for sender in agents:
            for receiver in agents:
                if sender == receiver:
                    continue

                dist = np.linalg.norm(pos_fn[sender] - pos_fn[receiver])
                if dist <= threshold:
                    self.assertTrue(ac.can_communicate(sender, receiver),
                                    "Sender and receiver communication are different")
                else:
                    self.assertFalse(ac.can_communicate(sender, receiver),
                                     "Sender and receiver communication is not possible")



