import unittest
from unittest.mock import  Mock


from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels

class TestDelayBasedModel(unittest.TestCase):
    """Message queue behavior tests"""

    def setUp(self):
        self.model = DelayBasedModel(['a0', 'a1'], min_delay=1, max_delay=3)
        self.model.rng = Mock()
        self.model.rng.integers.return_value = 2
        self.model.rng.random.return_value = 0.9

    def test_message_inserted_with_delay(self):
        """Inserts message with delay and success flag"""
        self.model._insert_message(0, 1)
        queue = self.model.message_queues[('a0', 'a1')]
        self.assertEqual(len(queue), 1)
        delay, success = queue[0]
        self.assertEqual(delay, 2)
        self.assertEqual(success, 1.0)
