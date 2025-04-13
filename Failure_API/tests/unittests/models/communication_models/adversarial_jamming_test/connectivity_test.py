from unittest.mock import Mock, patch
import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from collections import defaultdict
import numpy as np


class TestAdversarialJammingModelConnectivity(BaseAdversarialJammingModelTest):
    """"
    Test suite for AdversarialJammingModel connectivity matrix updates.
    """

    def test_update_connectivity_with_full_block(self):
        """Test update_connectivity with full blocking."""
        # Configure jamming for agent1->agent2
        self.mock_jammer_fn.return_value = True

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

        # Verify update was called with correct parameters
        mock_comms.update.assert_called_with('agent3', 'agent2', False)

        # Verify the jamming state was updated
        self.assertTrue(model.jamming_state[('agent3', 'agent2')])

        # Verify jamming log was updated
        self.assertEqual(model.jamming_log[('agent3', 'agent2')], [42])

    def test_update_connectivity_with_noise(self):
        """Test update_connectivity with signal degradation."""
        # jamming for agent1->agent2
        self.mock_jammer_fn.return_value = True

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn,
            full_block=False,
            noise_strength=0.3
        )

        # Create a mock ActiveCommunication object
        mock_comms = Mock(spec=ActiveCommunication)

        # Update connectivity
        model.update_connectivity(mock_comms)

        # Verify update was called with correct parameters
        mock_comms.update.assert_called_with('agent3', 'agent2', 0.3)

        # Verify the jamming state was updated
        self.assertTrue(model.jamming_state[('agent3', 'agent2')])

    def test_update_connectivity_with_float_jammer_result(self):
        """Test update_connectivity with a float jamming result."""
        # Configure jamming with strength 0.25
        self.mock_jammer_fn.return_value = 0.25

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn,
            full_block=False
        )

        # Create a mock ActiveCommunication object
        mock_comms = Mock(spec=ActiveCommunication)

        # Update connectivity
        model.update_connectivity(mock_comms)

        # Verify update was called with the jammer_fn returned value
        mock_comms.update.assert_called_with('agent3', 'agent2', 0.25)


if __name__ == '__main__':
    unittest.main()