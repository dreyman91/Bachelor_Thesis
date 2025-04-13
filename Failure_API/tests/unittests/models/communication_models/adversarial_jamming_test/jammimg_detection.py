""" This test reveals how the model decides which communication link to jam."""
from unittest.mock import Mock, patch
import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from collections import defaultdict
import numpy as np


class TestAJMJammingDetection(BaseAdversarialJammingModelTest):
    def setUp(self):
        super().setUp()

    def test_is_jammed_with_jammer_fn(self):
        """ Test is_jammed method with a custom jammer function."""

        self.mock_jammer_fn.return_value = True

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn
        )

        context = {
            'positions': self.mock_pos_fn(),
            'time': self.mock_time_fn()
        }
        #check jamming from agent1 to agent2
        result = model._is_jammed('agent1', 'agent2', context)

        self.assertTrue(result)

        args, kwargs = self.mock_jammer_fn.call_args
        self.assertEqual(args[0], 'agent1')
        self.assertEqual(args[1], 'agent2')
        np.testing.assert_array_equal(args[2], np.array([0, 0, 10]))
        np.testing.assert_array_equal(args[3], np.array([10, 0, 0]))
        self.assertEqual(args[4], 42)

    def test_is_jammed_with_schedule(self):
        """Test _is_jammed method with a schedule-based jamming."""
        # Configure the schedule function to activate jamming
        self.mock_jam_schedule_fn.return_value = True

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jam_schedule_fn=self.mock_jam_schedule_fn
        )
        model.set_rng(self.mock_rng.return_value)

        # Check jamming with an empty context
        result = model._is_jammed('agent1', 'agent2', {})

        # Verify the result and that the function was called
        self.assertTrue(result)
        self.mock_jam_schedule_fn.assert_called_once()

    def test_is_jammed_with_zone(self):
        """Test _is_jammed method with zone-based jamming."""
        # Configure the zone function to put agent2 in a jammed zone
        self.mock_jam_zone_fn.return_value = {
            'agent1': False,
            'agent2': True,
            'agent3': False
        }

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jam_zone_fn=self.mock_jam_zone_fn
        )
        model.set_rng(self.mock_rng.return_value)

        # Check jamming from agent1 to agent2 (agent2 is in jammed zone)
        result = model._is_jammed('agent1', 'agent2', {})

        # Verify the result and that the function was called
        self.assertTrue(result)
        self.mock_jam_zone_fn.assert_called_once()

        # Reset the mock and check jamming between non-jammed agents
        self.mock_jam_zone_fn.reset_mock()
        self.mock_jam_zone_fn.return_value = {
            'agent1': False,
            'agent2': False,
            'agent3': False
        }

        result= model._is_jammed('agent1', 'agent3', {})

        # Verify the result and that the function was called
        self.assertFalse(result)
        self.mock_jam_zone_fn.assert_called_once()

    def test_is_jammed_with_targeted_agents(self):
        """Test _is_jammed method with targeted agents."""
        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            targeted_agents=['agent2']
        )

        model.set_rng(self.mock_rng.return_value)

        # Check jamming from agent1 to agent2 (agent2 is targeted)
        result = model._is_jammed('agent1', 'agent2', {})

        # Verify agent2 is targeted (as receiver)
        self.assertTrue(result)

        # Check jamming from agent2 to agent3 (agent2 is targeted)
        result = model._is_jammed('agent2', 'agent3', {})

        # Verify agent2 is targeted (as sender)
        self.assertTrue(result)

        # Check jamming between non-targeted agents
        result = model._is_jammed('agent1', 'agent3', {})

        # Verify non-targeted agents are not jammed
        self.assertFalse(result)

    def test_is_jammed_with_previous_state(self):
        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jam_schedule_fn=self.mock_jam_schedule_fn
        )

        # Set up previous jamming state
        model.jamming_state[('agent1', 'agent2')] = True

        # Patch the _is_jammed method
        with patch.object(model, '_is_jammed') as mock_is_jammed:
            # Configure it to return True first, then False
            mock_is_jammed.side_effect = [True, False]

            # First call should be jammed
            result1 = mock_is_jammed('agent1', 'agent2', {})
            self.assertTrue(result1)

            # Second call should not be jammed
            result2 = mock_is_jammed('agent1', 'agent2', {})
            self.assertFalse(result2)


if __name__ == '__main__':
    unittest.main()