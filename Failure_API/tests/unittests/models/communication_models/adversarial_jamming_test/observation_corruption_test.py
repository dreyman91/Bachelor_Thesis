from unittest.mock import Mock, patch
import unittest
from.base_test import BaseAdversarialJammingModelTest
from Failure_API.src.failure_api.communication_models.adversarial_jamming_model import AdversarialJammingModel
from collections import defaultdict
import numpy as np


class TestAdversarialJammingModelObservations(BaseAdversarialJammingModelTest):
    """Test suite for AdversarialJammingModel observation corruption."""

    def test_get_corrupted_obs_with_full_block(self):
        """Test get_corrupted_obs with full blocking enabled."""
        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            full_block=True
        )

        # Original observation
        obs = {
            'agent1': np.array([1, 2, 3]),
            'agent2': np.array([4, 5, 6])
        }

        # Get corrupted observation for agent1
        result = model.get_corrupted_obs('agent1', obs)

        # With full_block=True, should return the original obs
        self.assertIs(result, obs)

    def test_get_corrupted_obs_with_noise(self):
        """Test get_corrupted_obs with noise application."""
        # Configure noise generation
        self.mock_normal.return_value = np.array([0.1, 0.2, 0.3])

        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            full_block=False,
            noise_strength=0.5
        )
        model.rng = Mock
        model.rng.normal = self.mock_normal

        # Original observation with numpy arrays
        obs = {
            'agent1': np.array([1, 2, 3]),
            'agent2': np.array([4, 5, 6])
        }

        # Get corrupted observation for agent2
        result = model.get_corrupted_obs('agent2', obs)

        # Verify self-observation is unchanged
        np.testing.assert_array_equal(result['agent2'], obs['agent2'])

        # Verify other observation is corrupted with noise
        np.testing.assert_array_equal(
            result['agent1'],
            np.array([1.1, 2.2, 3.3])  # Original + noise
        )

        # Verify noise was generated with correct parameters
        self.mock_normal.assert_called_with(0, 0.5, size=(3,))

    def test_get_corrupted_obs_with_mixed_types(self):
        """Test get_corrupted_obs with mixed data types."""
        model = AdversarialJammingModel(
            agent_ids=self.agent_ids,
            full_block=False,
            noise_strength=0.5
        )

        model.rng = Mock
        model.rng.normal = self.mock_normal

        # Original observation with mixed types
        obs = {
            'agent1': np.array([1, 2, 3]),
            'agent2': "message",
            'agent3': 42
        }

        # Get corrupted observation for agent1
        result = model.get_corrupted_obs('agent1', obs)

        # Verify non-array values are unchanged
        self.assertEqual(result['agent2'], "message")
        self.assertEqual(result['agent3'], 42)


if __name__ == '__main__':
    unittest.main()