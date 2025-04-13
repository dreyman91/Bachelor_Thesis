""""
Adversarial Jamming model simulates these forms of jamming in MARL:
- Temporal (time-based) jamming
- Spatial (poisiton-based) jamming
- Targeted (specific agents) jamming

Tests:

1. Test Initialization
2. Test the core Jamming logic
3. Verify that model properly updates the communication matrix.
4. Test how observations are corrupted by noise
5. Check that internal states tracking works properly
5. Test the pipeline (from jamming detection to output)
"""

import unittest
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from collections import defaultdict
import numpy as np
from unittest.mock import MagicMock, Mock, patch

class BaseAdversarialJammingModelTest(unittest.TestCase):
    """
    Comprehensive test suite for the AdversarialJammingModel class.

    Covers:
         1. Initialization and parameter validation
        2. Jamming detection mechanisms
        3. Connectivity matrix updates
        4. Observation corruption
        5. State management
        6. Integration tests
    """

    def setUp(self):

        self.agent_ids = ['agent1', 'agent2', 'agent3']

        # Mock functions to be passed to the model
        self.mock_pos_fn = Mock(return_value={
            'agent1': np.array([0, 0, 10]),
            'agent2': np.array([10, 0, 0]),
            'agent3': np.array([0, 10, 0]),
        })

        self.mock_time_fn = Mock(return_value=42)
        self.mock_jam_schedule_fn = Mock(return_value=False)

        self.mock_jam_zone_fn = Mock(return_value={
            'agent1': False,
            'agent2': False,
            'agent3': False,
        })

        self.mock_jammer_fn = Mock(return_value=False)

        self.rng_patcher = patch('numpy.random.RandomState')
        self.mock_rng = self.rng_patcher.start()

        # Mock RNG
        self.mock_rng.return_value.random = Mock(return_value = 0.5)

        self.mock_normal = Mock(return_value=np.zeros((3,)))
        self.mock_rng.return_value.normal = self.mock_normal


    def tearDown(self):
        """Clean up test after each process"""
        self.rng_patcher.stop()
        self.mock_pos_fn.reset_mock()
        self.mock_time_fn.reset_mock()
        self.mock_jam_schedule_fn.reset_mock()
        self.mock_jam_zone_fn.reset_mock()
        self.mock_jammer_fn.reset_mock()