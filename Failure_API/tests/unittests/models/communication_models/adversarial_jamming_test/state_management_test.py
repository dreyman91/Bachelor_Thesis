from unittest.mock import Mock, patch
import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from collections import defaultdict
import numpy as np


class TestAdversarialJammingModelStateManagement(BaseAdversarialJammingModelTest):
    """Test suite for AdversarialJammingModel state management."""

    def test_reset(self):
        """Test the reset method clears appropriate state."""
        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            time_fn=self.mock_time_fn
        )

        # Add some jamming log entries
        model.jamming_log[('agent1', 'agent2')] = [1, 2, 3]
        model.jamming_log[('agent2', 'agent3')] = [4, 5, 6]

        # Reset the model
        model.reset()

        # Verify the jamming log was cleared
        self.assertEqual(len(model.jamming_log), 0)

    def selective_jamming(self, sender, receiver, *args):
        return sender == "agent1" and receiver == "agent2"

    def test_update_jamming_states(self):
        """Test the _update_jamming_states method."""
        # Configure jamming for agent1->agent2
        self.mock_jammer_fn.side_effect = self.selective_jamming

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn
        )

        # Update jamming states
        model._update_jamming_states()

        # Verify the state was updated for agent1->agent2
        self.assertTrue(model.jamming_state[('agent1', 'agent2')])

        # Verify the state was not updated for other links
        self.assertFalse(model.jamming_state[('agent1', 'agent3')])
        self.assertFalse(model.jamming_state[('agent2', 'agent1')])
        self.assertFalse(model.jamming_state[('agent2', 'agent3')])
        self.assertFalse(model.jamming_state[('agent3', 'agent1')])
        self.assertFalse(model.jamming_state[('agent3', 'agent2')])


if __name__ == '__main__':
    unittest.main()