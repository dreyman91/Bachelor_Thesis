import  unittest
import sys

from .base_test import BaseAdversarialJammingModelTest
from .integration_test import TestAdversarialJammingModelIntegration
from .connectivity_test import TestAdversarialJammingModelConnectivity
from .initialization_test import AJMInitialization
from .jammimg_detection import TestAJMJammingDetection
from .state_management_test import TestAdversarialJammingModelStateManagement
from .observation_corruption_test import TestAdversarialJammingModelObservations


test_categories = {
    'init': AJMInitialization,
    'jam': TestAJMJammingDetection,
    'conn': TestAdversarialJammingModelConnectivity,
    'obs': TestAdversarialJammingModelObservations,
    'state': TestAdversarialJammingModelStateManagement,
    'integration': TestAdversarialJammingModelIntegration,
    'all': None
}

def print_usage():
    """Print usage instructions."""
    print("Usage: python run_ajm_tests.py <category>")
    print("Available categories:")
    print("  init       - Initialization tests")
    print("  jam        - Jamming detection tests")
    print("  conn       - Connectivity matrix tests")
    print("  obs        - Observation corruption tests")
    print("  state      - State management tests")
    print("  integration - Integration and edge case tests")
    print("  all        - All tests")

def run_tests(category):
    """ Run tests for the categories"""
    if category == 'all':
        suite = unittest.TestSuite()
        for test_class in test_categories.values():
            if test_class is not None:
                suite.addTest(unittest.makeSuite(test_class))

    else:
        # Run test for a specific category
        test_class = test_categories.get(category)
        if test_class is not None:
            print(f"Error: Unknown Category", {category})
            print_usage()
            return 1

        # Run the tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Return non-zero exit code if tests failed
        return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in test_categories:
        print_usage()
        sys.exit(1)

    category = sys.argv[1]
    sys.exit(run_tests(category))

