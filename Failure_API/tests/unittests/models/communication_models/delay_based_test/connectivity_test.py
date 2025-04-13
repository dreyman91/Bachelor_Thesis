import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from collections import defaultdict

from Failure_API.src.failure_api.communication_models.delay_based_model import DelayBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication


class TestDelayBasedModel(unittest.TestCase):
    """Connectivity matrix and communication update tests"""

    def setUp(self):
        self.agent_ids = ['agent_0', 'agent_1']
        self.model = DelayBasedModel(self.agent_ids, 1, 2)

        # ✅ Fully mock rng with random and integers
        mock_rng = MagicMock()
        mock_rng.random.return_value = 1.1
        mock_rng.integers.return_value = 1.0
        self.model.rng = mock_rng

    def test_skip_self_communication(self):
        """Self-communication should not insert messages"""
        matrix = ActiveCommunication(agent_ids=self.agent_ids)
        matrix.matrix = np.ones((2, 2))
        np.fill_diagonal(matrix.matrix, 1.0)
        self.model.update_connectivity(matrix)
        self.assertNotIn(('agent_0', 'agent_0'), self.model.message_queues)
        self.assertNotIn(('agent_1', 'agent_1'), self.model.message_queues)

    @patch('Failure_API.src.failure_api.communication_models.delay_based_model.ActiveCommunication.update')
    def test_message_delivered_when_delay_zero(self, mock_update):
        """Should call matrix.update when message delay reaches zero"""
        matrix = ActiveCommunication(agent_ids=self.agent_ids)
        matrix.matrix[:, :] = 1.0

        # Simulate proper message with delay = 1
        self.model._insert_message(0, 1)

        # Update should trigger delivery
        self.model.update_connectivity(matrix)

        mock_update.assert_called_with('agent_0', 'agent_1', 1.0)
        self.assertFalse(
            self.model.message_queues.get(('agent_0', 'agent_1')),
            msg="Message queue should be empty after delivery"
        )