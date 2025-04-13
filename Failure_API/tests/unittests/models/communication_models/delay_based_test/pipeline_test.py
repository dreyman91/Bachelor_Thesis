import unittest
from unittest.mock import Mock
from collections import deque
import numpy as np

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication

class TestDelayBasedModel(unittest.TestCase):
    """End-to-end communication simulation"""

    def setUp(self):
        self.agent_ids = ['agent_0', 'agent_1']
        self.model = DelayBasedModel(self.agent_ids, 1, 2)

        mock_rng = Mock()
        mock_rng.integers.return_value = 1
        mock_rng.random.return_value = 0.9
        self.model.rng = mock_rng

    def test_pipeline_over_multiple_steps(self):
        """Simulates multi-step communication with message delivery"""
        matrix = ActiveCommunication(agent_ids=self.agent_ids)
        matrix.matrix[:, :] = 1.0  # Full communication

        # Step 1: Insert initial messages
        self.model.update_connectivity(matrix)
        initial_queue = list(self.model.message_queues.get(('agent_0', 'agent_1'), []))
        self.assertEqual(len(initial_queue), 1)
        self.assertEqual(initial_queue[0][0], 1)  # delay

        # Step 2: Run update, should deliver old and insert new
        matrix.update = Mock()  # Track if it gets called
        self.model.update_connectivity(matrix)

        # Check that update was called to deliver the original message
        matrix.update.assert_any_call('agent_0', 'agent_1', 1.0)

        # Confirm a new message with delay 1 was inserted
        new_queue = list(self.model.message_queues.get(('agent_0', 'agent_1'), []))
        self.assertEqual(len(new_queue), 1)
        self.assertEqual(new_queue[0][0], 1)

