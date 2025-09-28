"""
Load Testing Execution Script and Setup Helper

This script provides easy setup and execution of load testing for the 
YouTube Recommendation System. It handles dependency installation,
environment verification, and test execution.

Usage:
    python run_load_tests.py --help
    python run_load_tests.py --setup
    python run_load_tests.py --quick
    python run_load_tests.py --scenario stress_test
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path and load environment variables
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # Change to project root directory

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"✅ Environment variables loaded from {project_root}/.env")
except ImportError:
    print("⚠️  python-dotenv not available, environment variables may not be loaded")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    required_packages = [
        'locust',
        'psutil', 
        'matplotlib',
        'seaborn',
        'faker',
        'requests'
    ]
    
    status = {}
    for package in required_packages:
        try:
            __import__(package)
            status[package] = True
        except ImportError:
            status[package] = False
    
    return status

def install_dependencies():
    """Install required dependencies for load testing"""
    print("📦 Installing load testing dependencies...")
    
    requirements_file = Path(__file__).parent / "load_testing_requirements.txt"
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def verify_system_running(host: str = "http://localhost:8080") -> bool:
    """Verify that the system under test is running"""
    try:
        import requests
        response = requests.get(f"{host}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ System is running at {host}")
            return True
        else:
            print(f"⚠️  System responded with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to system at {host}: {e}")
        return False

def setup_load_testing():
    """Complete setup for load testing"""
    print("🚀 Setting up load testing environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required for load testing")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check dependencies
    deps = check_dependencies()
    missing_deps = [pkg for pkg, installed in deps.items() if not installed]
    
    if missing_deps:
        print(f"📦 Missing dependencies: {', '.join(missing_deps)}")
        if not install_dependencies():
            return False
    else:
        print("✅ All dependencies are installed")
    
    # Create results directory
    results_dir = Path("load_test_results")
    results_dir.mkdir(exist_ok=True)
    print(f"📁 Results directory: {results_dir.absolute()}")
    
    # Create charts directory
    charts_dir = Path("load_test_charts")
    charts_dir.mkdir(exist_ok=True)
    print(f"📊 Charts directory: {charts_dir.absolute()}")
    
    # Verify system is running
    if verify_system_running():
        print("🎯 System verification passed")
    else:
        print("⚠️  Warning: Could not verify system is running")
        print("   Make sure your system is running on http://localhost:8080")
        print("   Or specify a different host with --host parameter")
    
    print("\n✅ Load testing setup complete!")
    print("\n📋 Quick start commands:")
    print("   python run_load_tests.py --quick")
    print("   python run_load_tests.py --scenario daily_traffic") 
    print("   python run_load_tests.py --suite comprehensive")
    
    return True

def run_load_test(scenario: Optional[str] = None, 
                 suite: Optional[str] = None,
                 host: str = "http://localhost:8080",
                 users: Optional[int] = None,
                 duration: Optional[int] = None,
                 verbose: bool = False) -> bool:
    """Run load test with specified parameters"""
    
    # Verify dependencies are installed
    deps = check_dependencies()
    missing_deps = [pkg for pkg, installed in deps.items() if not installed]
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Run: python run_load_tests.py --setup")
        return False
    
    # Verify system is running
    if not verify_system_running(host):
        print("❌ System is not running or not accessible")
        return False
    
    # Build command
    script_path = Path(__file__).parent / "load_test_scenarios.py"
    cmd = [sys.executable, str(script_path)]
    
    if scenario:
        cmd.extend(["--scenario", scenario])
    elif suite:
        cmd.extend(["--suite", suite])
    else:
        print("❌ Please specify either --scenario or --suite")
        return False
    
    cmd.extend(["--host", host])
    
    if users:
        cmd.extend(["--users", str(users)])
    if duration:
        cmd.extend(["--duration", str(duration)])
    
    print(f"🚀 Running load test...")
    if verbose:
        print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        # Run the load test
        result = subprocess.run(cmd, check=True)
        print("✅ Load test completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Load test failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Load test interrupted by user")
        return False

def list_scenarios():
    """List available load test scenarios"""
    scenarios = {
        "health_check": "Quick system responsiveness test (2 min, 10 users)",
        "daily_traffic": "Normal daily usage patterns (15 min, 100 users)",
        "peak_usage": "High traffic periods (20 min, 300 users)", 
        "stress_test": "Beyond normal capacity (30 min, 500 users)",
        "spike_test": "Sudden traffic increases (10 min, 200 users)",
        "endurance_test": "Sustained load over time (60 min, 150 users)",
        "database_focused": "Database performance testing (20 min, 100 users)",
        "recommendation_pipeline": "Full workflow testing (25 min, 200 users)",
        "user_feedback_heavy": "High feedback volume (15 min, 150 users)",
        "newsletter_generation": "Newsletter system load (10 min, 50 users)"
    }
    
    print("\n📋 Available Load Test Scenarios:")
    print("-" * 60)
    for name, description in scenarios.items():
        print(f"🎯 {name}")
        print(f"   {description}")
        print()

def list_suites():
    """List available test suites"""
    suites = {
        "quick": "Fast validation (~20 min) - health_check, daily_traffic",
        "standard": "Regular testing (~60 min) - health_check, daily_traffic, peak_usage, database_focused",
        "comprehensive": "Full evaluation (~2 hours) - All main scenarios",
        "performance": "Performance analysis (~2.5 hours) - daily_traffic, peak_usage, stress_test, endurance_test",
        "stability": "Stability testing (~1.5 hours) - endurance_test, spike_test, user_feedback_heavy"
    }
    
    print("\n📦 Available Test Suites:")
    print("-" * 60)
    for name, description in suites.items():
        print(f"📋 {name}")
        print(f"   {description}")
        print()

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Load Testing Setup and Execution for YouTube Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_load_tests.py --setup                    # Setup load testing environment
  python run_load_tests.py --quick                    # Run quick test suite
  python run_load_tests.py --scenario stress_test     # Run stress test
  python run_load_tests.py --suite comprehensive      # Run full test suite
  python run_load_tests.py --list-scenarios           # List available scenarios
  python run_load_tests.py --check                    # Check system status
        """
    )
    
    # Setup commands
    parser.add_argument("--setup", action="store_true", 
                       help="Install dependencies and setup load testing environment")
    parser.add_argument("--check", action="store_true",
                       help="Check dependencies and system status")
    
    # Test execution
    parser.add_argument("--scenario", help="Run specific load test scenario")
    parser.add_argument("--suite", help="Run test suite (quick, standard, comprehensive, performance, stability)")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite")
    
    # Test parameters
    parser.add_argument("--host", default="http://localhost:8080",
                       help="Target host for load testing (default: http://localhost:8080)")
    parser.add_argument("--users", type=int, help="Override maximum number of users")
    parser.add_argument("--duration", type=int, help="Override test duration in minutes")
    
    # Information commands
    parser.add_argument("--list-scenarios", action="store_true", 
                       help="List available load test scenarios")
    parser.add_argument("--list-suites", action="store_true",
                       help="List available test suites")
    
    # Options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Handle information commands
    if args.list_scenarios:
        list_scenarios()
        return
    
    if args.list_suites:
        list_suites()
        return
    
    # Handle setup command
    if args.setup:
        success = setup_load_testing()
        sys.exit(0 if success else 1)
    
    # Handle check command
    if args.check:
        print("🔍 Checking load testing environment...")
        
        deps = check_dependencies()
        missing_deps = [pkg for pkg, installed in deps.items() if not installed]
        
        if missing_deps:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            print("   Run: python run_load_tests.py --setup")
        else:
            print("✅ All dependencies installed")
        
        verify_system_running(args.host)
        return
    
    # Handle test execution
    if args.quick:
        success = run_load_test(suite="quick", host=args.host, 
                              users=args.users, duration=args.duration, verbose=args.verbose)
    elif args.scenario:
        success = run_load_test(scenario=args.scenario, host=args.host,
                              users=args.users, duration=args.duration, verbose=args.verbose)
    elif args.suite:
        success = run_load_test(suite=args.suite, host=args.host,
                              users=args.users, duration=args.duration, verbose=args.verbose)
    else:
        print("❓ No action specified. Use --help for usage information.")
        print("\n🚀 Quick start:")
        print("   python run_load_tests.py --setup      # First time setup")
        print("   python run_load_tests.py --quick       # Run quick tests")
        print("   python run_load_tests.py --list-scenarios  # See all scenarios")
        return
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()