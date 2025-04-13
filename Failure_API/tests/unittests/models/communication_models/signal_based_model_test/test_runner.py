
"""
Test Runner for SignalBasedModel Tests

This script serves as a central entry point for running all the test files
related to the SignalBasedModel. It discovers and runs all test files in the
current directory that follow the naming pattern 'test_signal_*.py'.

Usage:
    python test_runner.py

This approach allows us to:
1. Run all tests with a single command
2. Keep individual test files focused on specific aspects of functionality
3. Add new test files easily without modifying the runner
4. Control verbosity and other test execution parameters centrally

Compatible with the PettingZoo framework testing approach.
"""

import unittest
import sys
import os


def run_all_tests():
    """
    Discover and run all test files for the SignalBasedModel.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Create a test loader
    loader = unittest.TestLoader()

    # Define the test discovery pattern
    start_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = 'test_*.py'

    # Discover all tests matching the pattern
    test_suite = loader.discover(start_dir, pattern=pattern)

    # Create a test runner with verbosity level 2 (more detailed output)
    runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    result = runner.run(test_suite)

    # Return True if all tests passed, False otherwise
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run the tests and set exit code based on success/failure
    success = run_all_tests()
    sys.exit(0 if success else 1)