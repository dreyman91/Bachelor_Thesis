import unittest
from collections import deque

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestDelayBasedModel(unittest.TestCase):
    """State management and reset tests"""

    def setUp(self):
        self.model = DelayBasedModel(['a0', 'a1'], 1, 3)

    def test_reset_clears_queues(self):
        """Reset should empty message queues"""
        self.model.message_queues[('a0', 'a1')] = deque([(2, 1.0)])
        self.model.reset()
        self.assertEqual(len(self.model.message_queues), 0)
