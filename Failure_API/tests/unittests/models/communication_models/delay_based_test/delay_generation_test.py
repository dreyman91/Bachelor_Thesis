import unittest
from unittest.mock import  Mock, patch, MagicMock
import numpy as np
from collections import deque
from parameterized import parameterized
import warnings

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels


class TestDelayBasedModel(unittest.TestCase):
    """Tests for Delay Generation Logic"""

    def setUp(self):
        self.agent_ids = ['a0', 'a1']
        self.model = DelayBasedModel(agent_ids=self.agent_ids, min_delay=2, max_delay=5)
        self.model.rng = Mock()
        self.model.rng.integers.return_value = 3

    def test_random_delay_in_bounds(self):
        """Should generate delay within [min_delay, max_delay]"""
        delay = self.model._generate_delay('a0', 'a1')
        self.assertGreaterEqual(delay, self.model.min_delay)
        self.assertLessEqual(delay, self.model.max_delay)

    def test_delay_with_time_fn(self):
        """Should add +2 if time_fn returns multiple of 15."""
        self.model.time_fn = Mock(return_value=30)
        delay = self.model._generate_delay('a0', 'a1')
        self.assertEqual(delay, min(3 + 2, self.model.max_delay))

    def test_delay_with_traffic_fn(self):
        """Should add delay based on traffic volume."""
        self.model.traffic_fn = Mock(return_value={('a0', 'a1'): 0.5})
        delay = self.model._generate_delay('a0', 'a1')
        self.assertEqual(delay, min(3 + int(0.5 * 3), self.model.max_delay))

    def test_custom_delay_fn(self):
        """Should use delay_fn if defined."""
        self.model.delay_fn = Mock(return_value=4)
        delay = self.model._generate_delay('a0', 'a1')
        self.assertEqual(delay, 4)

