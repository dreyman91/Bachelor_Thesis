"""
Script to run all test cases for the ContextAwareMarkovModel.

This script discovers and runs all test modules for the ContextAwareMarkovModel class,
providing a consolidated test report.
"""

import unittest
import sys
import os
import argparse


def run_all_tests(verbosity=2, pattern='test_*.py', performance=True):
    """
    Discover and run all tests for the ContextAwareMarkovModel.

    Args:
        verbosity (int): Test output verbosity level (1-3)
        pattern (str): File pattern to match test files
        performance (bool): Whether to run performance tests

    Returns:
        unittest.TestResult: The test result object
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create test loader
    loader = unittest.TestLoader()

    # Discover all tests in the current directory matching the pattern
    test_suite = loader.discover(current_dir, pattern=pattern)

    # If performance tests should be skipped, modify the suite
    if not performance:
        # Create a new filtered suite
        filtered_suite = unittest.TestSuite()

        # Helper function to filter out performance tests
        def add_if_not_performance(test_case):
            if hasattr(test_case, '_testMethodName'):
                if test_case._testMethodName.startswith('test_performance'):
                    # Set a flag to skip the test
                    setattr(test_case, 'skipLongTests', True)
                filtered_suite.addTest(test_case)
            else:
                # This is a test suite, recurse
                for test in test_case:
                    add_if_not_performance(test)

        # Filter the discovered tests
        for test in test_suite:
            add_if_not_performance(test)

        test_suite = filtered_suite

    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=verbosity)

    # Run all tests
    result = runner.run(test_suite)

    return result


if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run all tests for ContextAwareMarkovModel')

    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[1, 2, 3],
        default=2,
        help='Test output verbosity (1-3)'
    )

    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='test_*.py',
        help='File pattern to match test files'
    )

    parser.add_argument(
        '--skip-performance',
        action='store_true',
        help='Skip performance tests'
    )

    # Parse arguments
    args = parser.parse_args()

    # Run tests with the specified options
    result = run_all_tests(
        verbosity=args.verbosity,
        pattern=args.pattern,
        performance=not args.skip_performance
    )

    # Set exit code based on test results
    sys.exit(not result.wasSuccessful())