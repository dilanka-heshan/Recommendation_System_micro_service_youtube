"""
Configuration Testing Suite Runner
Run configuration tests for the YouTube Recommendation System
"""
import pytest
import sys
import os
from pathlib import Path

def run_configuration_tests():
    """Run all configuration tests"""
    
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Configuration test directory
    config_test_dir = Path(__file__).parent
    
    print("=" * 60)
    print("CONFIGURATION TESTING SUITE")
    print("=" * 60)
    print(f"Running tests from: {config_test_dir}")
    print()
    
    # Test categories to run
    test_categories = [
        ("Database Configuration", "test_database_config.py"),
        ("Environment Variables", "test_environment_config.py"),
        ("Service Configuration", "test_service_config.py"),
        ("Integration Scenarios", "test_integration_config.py")
    ]
    
    results = {}
    
    for category_name, test_file in test_categories:
        print(f"Running {category_name} Tests...")
        print("-" * 40)
        
        test_path = config_test_dir / test_file
        
        # Run pytest for each test file
        exit_code = pytest.main([
            str(test_path),
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "-x"  # Stop on first failure
        ])
        
        results[category_name] = "PASSED" if exit_code == 0 else "FAILED"
        print(f"{category_name}: {results[category_name]}")
        print()
    
    # Summary
    print("=" * 60)
    print("CONFIGURATION TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for result in results.values() if result == "PASSED")
    total_count = len(results)
    
    for category, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {category}: {result}")
    
    print()
    print(f"Overall: {passed_count}/{total_count} test categories passed")
    
    if passed_count == total_count:
        print("🎉 All configuration tests passed!")
        return 0
    else:
        print("❌ Some configuration tests failed!")
        return 1

def run_quick_config_tests():
    """Run a subset of quick configuration tests"""
    config_test_dir = Path(__file__).parent
    
    print("Running Quick Configuration Tests...")
    
    # Run only basic tests
    exit_code = pytest.main([
        str(config_test_dir / "test_database_config.py::TestMongoDBConfiguration::test_mongodb_with_valid_connection_string"),
        str(config_test_dir / "test_environment_config.py::TestEnvironmentVariableValidation::test_required_environment_variables"),
        str(config_test_dir / "test_service_config.py::TestFastAPIConfiguration::test_health_check_endpoint"),
        "-v"
    ])
    
    return exit_code

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run configuration tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--category", help="Run specific test category")
    
    args = parser.parse_args()
    
    if args.quick:
        exit_code = run_quick_config_tests()
    elif args.category:
        # Run specific category
        config_test_dir = Path(__file__).parent
        test_file = f"test_{args.category.lower()}_config.py"
        exit_code = pytest.main([str(config_test_dir / test_file), "-v"])
    else:
        exit_code = run_configuration_tests()
    
    sys.exit(exit_code)