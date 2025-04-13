import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    tests = loader.discover(start_dir=".", pattern="*_test.py")
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(tests)
