#!/usr/bin/env python3
"""
Simple test runner similar to Go's 'go test' command.
Usage: python3 test_runner.py [pattern]
"""

import unittest
import sys
import os

def run_tests(pattern="*test*.py"):
    """Run tests with optional pattern matching"""
    # Discover tests in current directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover all test files
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_simple_tests():
    """Run the simple assert-based tests"""
    try:
        # Import and run the simple test
        from bpe_simple_test import run_all_tests
        return run_all_tests()
    except ImportError as e:
        print(f"Could not import simple tests: {e}")
        return False

def main():
    """Main function - similar to go test command"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "simple":
            # Run simple assert-based tests
            success = run_simple_tests()
        elif sys.argv[1] == "unittest":
            # Run unittest-based tests
            success = run_tests("bpe_test.py")
        elif sys.argv[1] == "all":
            # Run both
            success1 = run_simple_tests()
            success2 = run_tests("bpe_test.py")
            success = success1 and success2
        else:
            # Run with custom pattern
            success = run_tests(sys.argv[1])
    else:
        # Default: run simple tests
        success = run_simple_tests()
    
    # Exit with appropriate code (like go test)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
