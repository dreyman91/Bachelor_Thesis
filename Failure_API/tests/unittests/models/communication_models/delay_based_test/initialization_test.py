import unittest
import numpy as np
import warnings

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels

class TestDelayBasedModel(unittest.TestCase):
    """Tests for DelayBasedModel Initialization"""

    def setUp(self):
        self.agent_ids = ['agent_0', 'agent_1']

    def test_valid_initialization(self):
        """Should initialize with proper default attributes."""
        model = DelayBasedModel(agent_ids=self.agent_ids, min_delay=1, max_delay=5)
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertEqual(model.min_delay, 1)
        self.assertEqual(model.max_delay, 5)
        self.assertIsNotNone(model.message_queues)

    def test_negative_delay_values(self):
        """Should clip negative delay values to 0."""
        model = DelayBasedModel(agent_ids=self.agent_ids, min_delay=-2, max_delay=-1)
        self.assertEqual(model.min_delay, 0)
        self.assertEqual(model.max_delay, 0)

    def test_min_delay_greater_than_max(self):
        """Should swap delay values if min_delay > max_delay."""
        model = DelayBasedModel(agent_ids=self.agent_ids, min_delay=7, max_delay=3)
        self.assertEqual(model.min_delay, 3)
        self.assertEqual(model.max_delay, 7)

    def test_empty_agent_list(self):
        """Should initialize with no errors on empty agent list."""
        model = DelayBasedModel(agent_ids=[], min_delay=0, max_delay=2)
        self.assertEqual(model.agent_ids, [])
