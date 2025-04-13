import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from collections import defaultdict
import numpy as np
from unittest.mock import Mock, patch


class AJMInitialization(BaseAdversarialJammingModelTest):
    """
     Test suite for AJMInitialization class.
     """

    def setUp(self):
        super().setUp()
    def basic_initialization(self):

        model = AdversarialJammingModel(agent_ids=self.agent_ids)

        # Verify model attributes
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertTrue(model.full_block)
        self.assertEqual(model.noise_strength, 0.0)
        self.assertEqual(model.targeted_agents, set())

        # Verify default probability values
        self.assertTrue(model.p_stay_jammed, 0.8)
        self.assertEqual(model.p_recover, 0.2)
        self.assertEqual(model.p_start_jam, 0.1)

        # Verify state initialization
        self.assertIsInstance(model.jamming_state, defaultdict)
        self.assertIsInstance(model.jamming_log, defaultdict)

    def test_initialization_with_all_parameters(self):
        targeted_agents = ['agent1']

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            jam_schedule_fn=self.mock_jam_schedule_fn,
            jam_zone_fn=self.mock_jam_zone_fn,
            targeted_agents=targeted_agents,
            pos_fn=self.mock_pos_fn,
            time_fn=self.mock_time_fn,
            full_block=False,
            noise_strength=0.3,
            p_stay=0.7,
            p_recover=0.3,
            p_start=0.2
        )

        # Verify custom parameters
        self.assertEqual(model.agent_ids, self.agent_ids)
        self.assertFalse(model.full_block)
        self.assertEqual(model.noise_strength, 0.3)
        self.assertEqual(model.targeted_agents, set(targeted_agents))

        # Verify custom probability values
        self.assertEqual(model.p_stay_jammed, 0.7)
        self.assertEqual(model.p_recover, 0.3)
        self.assertEqual(model.p_start_jam, 0.2)

        # Verify function assignments
        self.assertEqual(model.jammer_fn, self.mock_jammer_fn)
        self.assertEqual(model.jam_schedule_fn, self.mock_jam_schedule_fn)
        self.assertEqual(model.jam_zone_fn, self.mock_jam_zone_fn)
        self.assertEqual(model.pos_fn, self.mock_pos_fn)
        self.assertEqual(model.time_fn, self.mock_time_fn)

    @patch('builtins.print')
    def test_initialization_warnings(self, mock_print):
        """ Test that appropriate warnings are issued during initialization."""

        #Test warning for contradictory options
        AdversarialJammingModel(
            agent_ids=self.agent_ids,
            jammer_fn=self.mock_jammer_fn,
            jam_schedule_fn=self.mock_jam_schedule_fn,
        )

        mock_print.assert_called_with(
            "Warning: Both jammer_fn and simple jamming options provided. jammer_fn will take precedence"
        )
        mock_print.reset_mock()

        # Test warning for no jam mechanism
        # Test warning for no jamming mechanism
        AdversarialJammingModel(agent_ids=self.agent_ids)

        mock_print.assert_called_with(
            "Warning: No jamming mechanism specified. Model will have no effect."
        )


if __name__ == '__main__':
    unittest.main()