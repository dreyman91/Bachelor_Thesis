from unittest.mock import Mock, patch
import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from collections import defaultdict
import numpy as np


class TestAdversarialJammingModelIntegration(BaseAdversarialJammingModelTest):
    """Test suite for AdversarialJammingModel integration tests and edge cases."""

    def test_full_jamming_pipeline(self):
        """Test the full pipeline from jamming detection to matrix updates."""

        # Configure selective jamming
        def custom_jammer(sender, receiver, sender_pos, receiver_pos, time):
            # Jam only if sender is agent1 and receiver is agent2
            return sender == 'agent1' and receiver == 'agent2'

        self.mock_jammer_fn.side_effect = custom_jammer

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn,
            full_block=True
        )

        # Create a mock ActiveCommunication object
        mock_comms = Mock(spec=ActiveCommunication)

        # Update connectivity
        model.update_connectivity(mock_comms)

        # Verify the correct jamming pattern
        mock_comms.update.assert_called_with('agent1', 'agent2', False)
        self.assertEqual(mock_comms.update.call_count, 1)  # Only one link should be jammed

        # Verify the jamming state was updated correctly
        self.assertTrue(model.jamming_state[('agent1', 'agent2')])
        self.assertFalse(model.jamming_state.get(('agent1', 'agent3'), False))
        self.assertFalse(model.jamming_state.get(('agent2', 'agent3'), False))

    def test_create_initial_matrix(self):
        """Test the create_initial_matrix static method."""
        matrix = AdversarialJammingModel.create_initial_matrix(self.agent_ids)

        # Verify dimensions
        self.assertEqual(matrix.shape, (3, 3))

        # Verify diagonal is False
        self.assertFalse(matrix[0, 0])
        self.assertFalse(matrix[1, 1])
        self.assertFalse(matrix[2, 2])

        # Verify all other elements are True
        self.assertTrue(matrix[0, 1])
        self.assertTrue(matrix[0, 2])
        self.assertTrue(matrix[1, 0])
        self.assertTrue(matrix[1, 2])
        self.assertTrue(matrix[2, 0])
        self.assertTrue(matrix[2, 1])

    def test_error_handling_in_callbacks(self):
        """Test error handling in callback functions."""
        # Configure jammer_fn to raise an exception
        self.mock_jammer_fn.side_effect = Exception("Test exception")

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn
        )

        # Should not raise an exception, but log it
        with patch('builtins.print') as mock_print:
            result = model._is_jammed('agent1', 'agent2', {})

            # Verify the error was logged
            mock_print.assert_called_with(
                "Error in jammer_fn for agent1->agent2: Test exception"
            )

            # Verify the method returns False on error
            self.assertFalse(result)

    def test_empty_agent_list(self):
        """Test behavior with an empty agent list."""
        model = AdversarialJammingModel(agent_ids=[])

        # Create a mock ActiveCommunication object
        mock_comms = Mock(spec=ActiveCommunication)

        # Should not raise an exception
        model.update_connectivity(mock_comms)

        # Verify no updates were made
        mock_comms.update.assert_not_called()


if __name__ == '__main__':
    unittest.main()