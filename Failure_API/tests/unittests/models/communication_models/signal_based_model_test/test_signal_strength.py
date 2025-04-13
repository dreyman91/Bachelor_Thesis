
"""
Signal Strength Calculation Tests for SignalBasedModel

This test file focuses on testing the signal strength calculation functionality
of the SignalBasedModel class. The `_signal_strength` method implements an
inverse-square law with a softening term to compute signal strength based on
distance between agents.

We verify that:
1. The signal strength calculation follows the inverse-square law
2. The softening term prevents division by zero for very small distances
3. The method handles extreme distance values correctly
4. The transmit power parameter properly scales the signal strength
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import math
from typing import Dict, List

from Failure_API.src.failure_api.communication_models.signal_based_failure import SignalBasedModel
from Failure_API.src.failure_api.communication_models.active_communication import ActiveCommunication
from Failure_API.src.failure_api.communication_models.base_communication_model import CommunicationModels

class TestSignalStrengthCalculation(unittest.TestCase):
    """
    Test class for verifying the signal strength calculation of SignalBasedModel.

    These tests ensure that the `_signal_strength` method correctly calculates
    signal strength according to the inverse-square law with appropriate handling
    of edge cases.
    """

    def setUp(self):
        """
        Set up common test fixtures before each test.

        Creates a SignalBasedModel instance with standard parameters and
        defines some test distances to use across multiple tests.
        """
        # Create a simple agent list and mock position function
        self.agent_ids = ["agent1", "agent2"]
        self.pos_fn = Mock(return_value={
            "agent1": np.array([0.0, 0.0, 0.0]),
            "agent2": np.array([1.0, 0.0, 0.0])
        })

        # Create a model with default parameters
        self.model = SignalBasedModel(
            agent_ids=self.agent_ids,
            pos_fn=self.pos_fn
        )

        # Define some test distances to use across multiple tests
        self.test_distances = [0.0, 0.1, 1.0, 2.0, 10.0, 100.0]

        # Define the softening term used in the model (epsilon)
        self.epsilon = 1e-6

    def test_inverse_square_law(self):
        """
        Test that signal strength follows the inverse-square law.

        The signal strength should be proportional to 1/d² where d is the distance.
        This test verifies this relationship for various distances.
        """
        # For each test distance (except zero), verify inverse-square relationship
        for dist in self.test_distances:
            if dist > 0:  # Skip zero distance for this test
                expected_strength = self.model.tx_power / (dist ** 2 + self.epsilon)
                calculated_strength = self.model._signal_strength(dist)

                # Use assertAlmostEqual with a relative tolerance for floating point comparison
                self.assertAlmostEqual(
                    calculated_strength,
                    expected_strength,
                    delta=1e-10,
                    msg=f"Signal strength calculation incorrect for distance {dist}"
                )

    def test_zero_distance(self):
        """
        Test signal strength calculation with zero distance.

        Even at zero distance, the calculation should not result in division by zero
        due to the softening term (epsilon).
        """
        # Calculate signal strength at zero distance
        strength = self.model._signal_strength(0.0)

        # Expected strength is tx_power / epsilon
        expected_strength = self.model.tx_power / self.epsilon

        # Verify the calculation
        self.assertAlmostEqual(
            strength,
            expected_strength,
            delta=1e-10,
            msg="Signal strength calculation incorrect for zero distance"
        )

    def test_transmit_power_scaling(self):
        """
        Test that transmit power properly scales the signal strength.

        Signal strength should be linearly proportional to the transmit power.
        This test verifies this relationship by comparing calculations with
        different transmit power values.
        """
        # Test distance
        dist = 2.0

        # Calculate signal strength with default transmit power (1.0)
        base_strength = self.model._signal_strength(dist)

        # Create models with different transmit powers
        test_powers = [0.5, 2.0, 5.0, 10.0]

        for power in test_powers:
            # Create a model with the test power
            model = SignalBasedModel(
                agent_ids=self.agent_ids,
                pos_fn=self.pos_fn,
                tx_power=power
            )

            # Calculate signal strength with the test power
            strength = model._signal_strength(dist)

            # The ratio of strengths should equal the ratio of powers
            expected_ratio = power / self.model.tx_power
            actual_ratio = strength / base_strength

            # Verify the ratio
            self.assertAlmostEqual(
                actual_ratio,
                expected_ratio,
                delta=1e-10,
                msg=f"Signal strength scaling incorrect for power {power}"
            )

    def test_very_large_distances(self):
        """
        Test signal strength calculation with very large distances.

        For very large distances, the signal strength should approach zero,
        but should always be positive.
        """
        # Test with extremely large distances
        large_distances = [1e3, 1e6, 1e9]

        for dist in large_distances:
            strength = self.model._signal_strength(dist)

            # Verify that strength is positive
            self.assertGreater(
                strength,
                0.0,
                msg=f"Signal strength not positive for distance {dist}"
            )

            # Verify that strength approaches the expected value
            expected_strength = self.model.tx_power / (dist ** 2)

            # Use a relative tolerance for very small values
            relative_error = abs((strength - expected_strength) / expected_strength)
            self.assertLess(
                relative_error,
                1e-6,
                msg=f"Signal strength calculation inaccurate for large distance {dist}"
            )

    def test_very_small_distances(self):
        """
        Test signal strength calculation with very small distances.

        For very small distances, the softening term becomes important to prevent
        extremely large signal strength values.
        """
        # Test with extremely small distances
        small_distances = [1e-3, 1e-6, 1e-9]

        for dist in small_distances:
            strength = self.model._signal_strength(dist)

            # Calculate expected strength with the softening term
            expected_strength = self.model.tx_power / (dist ** 2 + self.epsilon)

            # Verify the calculation
            self.assertAlmostEqual(
                strength,
                expected_strength,
                delta=1e-10,
                msg=f"Signal strength calculation incorrect for small distance {dist}"
            )

    def test_negative_distances(self):
        """
        Test signal strength calculation with negative distances.

        Although negative distances don't make physical sense, the function should
        still handle them gracefully by squaring the distance.
        """
        # Test with negative distances
        negative_distances = [-1.0, -2.0, -10.0]

        for dist in negative_distances:
            # Signal strength for negative distance should equal that for positive distance
            neg_strength = self.model._signal_strength(dist)
            pos_strength = self.model._signal_strength(abs(dist))

            # Verify that they are equal
            self.assertEqual(
                neg_strength,
                pos_strength,
                msg=f"Signal strength calculation different for negative distance {dist}"
            )


if __name__ == "__main__":
    unittest.main()