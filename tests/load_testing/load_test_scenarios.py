"""
Load Testing Scenarios and Test Suites for YouTube Recommendation System

This module defines various load testing scenarios that simulate real-world usage patterns:
- Normal daily traffic patterns
- Peak usage scenarios (viral videos, trending topics)
- Stress testing (finding system breaking points)
- Spike testing (sudden traffic increases)
- Endurance testing (sustained load over time)

Usage:
    python load_test_scenarios.py --scenario daily_traffic
    python load_test_scenarios.py --scenario peak_usage
    python load_test_scenarios.py --scenario stress_test
"""

import argparse
import time
import json
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from performance_monitor import PerformanceMonitor, MonitoringConfig
except ImportError:
    print("⚠️  Performance monitor not available, using basic monitoring")
    class PerformanceMonitor:
        def __init__(self, config=None): pass
        def start_monitoring(self): pass
        def generate_report(self): return {"status": "basic_monitoring"}
    class MonitoringConfig: pass

try:
    from database_load_tests import DatabaseLoadTester, DatabaseLoadTestConfig
except ImportError:
    print("⚠️  Database load tests not available, database scenarios will be skipped")
    class DatabaseLoadTester:
        def __init__(self, config): pass
        def run_load_test(self): return {"status": "database_testing_unavailable"}
    class DatabaseLoadTestConfig: pass

@dataclass
class LoadTestScenario:
    """Defines a load testing scenario"""
    name: str
    description: str
    duration_minutes: int
    max_users: int
    spawn_rate: float  # users per second
    locust_file: str
    host: str = "http://localhost:8080"
    additional_params: Dict[str, Any] = None

class LoadTestRunner:
    """Manages execution of different load testing scenarios"""
    
    def __init__(self):
        self.scenarios = self._define_scenarios()
        self.results_dir = Path("load_test_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def _define_scenarios(self) -> Dict[str, LoadTestScenario]:
        """Define all available load testing scenarios"""
        return {
            "health_check": LoadTestScenario(
                name="Health Check",
                description="Quick health check to verify system is responsive",
                duration_minutes=2,
                max_users=10,
                spawn_rate=2.0,
                locust_file="api_endpoint_tests.py"
            ),
            
            "daily_traffic": LoadTestScenario(
                name="Daily Traffic Pattern",
                description="Simulates normal daily traffic patterns with gradual user increases",
                duration_minutes=15,
                max_users=100,
                spawn_rate=5.0,
                locust_file="api_endpoint_tests.py"
            ),
            
            "peak_usage": LoadTestScenario(
                name="Peak Usage",
                description="Simulates peak usage periods (trending videos, viral content)",
                duration_minutes=20,
                max_users=300,
                spawn_rate=15.0,
                locust_file="api_endpoint_tests.py"
            ),
            
            "stress_test": LoadTestScenario(
                name="Stress Test",
                description="Pushes system beyond normal capacity to find breaking points",
                duration_minutes=30,
                max_users=500,
                spawn_rate=25.0,
                locust_file="api_endpoint_tests.py",
                additional_params={"expect-workers": 4}
            ),
            
            "spike_test": LoadTestScenario(
                name="Spike Test",
                description="Sudden spike in traffic to test system resilience",
                duration_minutes=10,
                max_users=200,
                spawn_rate=50.0,  # Very rapid spawn rate
                locust_file="api_endpoint_tests.py"
            ),
            
            "endurance_test": LoadTestScenario(
                name="Endurance Test",
                description="Sustained moderate load over extended period",
                duration_minutes=60,
                max_users=150,
                spawn_rate=5.0,
                locust_file="api_endpoint_tests.py"
            ),
            
            "database_focused": LoadTestScenario(
                name="Database Focused Test",
                description="Tests database performance under various load conditions",
                duration_minutes=20,
                max_users=100,
                spawn_rate=10.0,
                locust_file="database_load_tests.py"
            ),
            
            "recommendation_pipeline": LoadTestScenario(
                name="Recommendation Pipeline Test",
                description="Focuses on the complete recommendation workflow",
                duration_minutes=25,
                max_users=200,
                spawn_rate=10.0,
                locust_file="api_endpoint_tests.py",
                additional_params={"tags": "recommendations,workflow"}
            ),
            
            "user_feedback_heavy": LoadTestScenario(
                name="User Feedback Heavy Load",
                description="Simulates high volume of user feedback and vector updates",
                duration_minutes=15,
                max_users=150,
                spawn_rate=12.0,
                locust_file="api_endpoint_tests.py",
                additional_params={"tags": "user-vector-update,feedback"}
            ),
            
            "newsletter_generation": LoadTestScenario(
                name="Newsletter Generation Load",
                description="Tests newsletter generation under load",
                duration_minutes=10,
                max_users=50,
                spawn_rate=5.0,
                locust_file="api_endpoint_tests.py",
                additional_params={"tags": "newsletter"}
            )
        }
    
    def run_scenario(self, scenario_name: str, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a specific load testing scenario"""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name]
        print(f"\n🚀 Starting Load Test Scenario: {scenario.name}")
        print(f"📝 Description: {scenario.description}")
        print(f"⏱️  Duration: {scenario.duration_minutes} minutes")
        print(f"👥 Max Users: {scenario.max_users}")
        print(f"📈 Spawn Rate: {scenario.spawn_rate} users/second")
        print("-" * 60)
        
        # Setup monitoring
        monitor_config = MonitoringConfig(
            collection_interval=1.0,
            enable_alerts=True,
            save_raw_data=True
        )
        monitor = PerformanceMonitor(monitor_config)
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Run the load test
            if scenario_name == "database_focused":
                results = self._run_database_test_scenario(scenario, custom_params)
            else:
                results = self._run_locust_scenario(scenario, custom_params)
            
            # Stop monitoring and get results
            monitoring_results = monitor.generate_report()
            
            # Combine results
            combined_results = {
                "scenario": {
                    "name": scenario.name,
                    "description": scenario.description,
                    "start_time": results.get("start_time"),
                    "end_time": results.get("end_time"),
                    "duration_minutes": scenario.duration_minutes
                },
                "load_test_results": results,
                "monitoring_results": monitoring_results,
                "recommendations": self._generate_scenario_recommendations(scenario, results, monitoring_results)
            }
            
            # Save results
            self._save_results(scenario_name, combined_results)
            
            # Print summary
            self._print_results_summary(combined_results)
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Error running scenario {scenario_name}: {e}")
            monitor.stop_monitoring()
            raise
    
    def _run_locust_scenario(self, scenario: LoadTestScenario, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a Locust-based load test scenario"""
        # Build Locust command
        cmd = [
            "locust",
            "-f", f"tests/load_testing/{scenario.locust_file}",
            "--host", scenario.host,
            "--users", str(scenario.max_users),
            "--spawn-rate", str(scenario.spawn_rate),
            "--run-time", f"{scenario.duration_minutes}m",
            "--headless",
            "--html", f"load_test_results/{scenario.name.lower().replace(' ', '_')}_report.html",
            "--csv", f"load_test_results/{scenario.name.lower().replace(' ', '_')}"
        ]
        
        # Add additional parameters
        if scenario.additional_params:
            for key, value in scenario.additional_params.items():
                cmd.extend([f"--{key}", str(value)])
        
        if custom_params:
            for key, value in custom_params.items():
                cmd.extend([f"--{key}", str(value)])
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        start_time = datetime.now()
        
        try:
            # Run Locust
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=scenario.duration_minutes * 60 + 300)  # 5 min buffer
            
            end_time = datetime.now()
            
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": "Timeout expired",
                "success": False
            }
    
    def _run_database_test_scenario(self, scenario: LoadTestScenario, custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run database-focused load test scenario"""
        config = DatabaseLoadTestConfig(
            test_duration_seconds=scenario.duration_minutes * 60,
            concurrent_connections=scenario.max_users,
            operations_per_second=int(scenario.spawn_rate * 10)  # Adjust for database operations
        )
        
        tester = DatabaseLoadTester(config)
        
        start_time = datetime.now()
        
        try:
            results = tester.run_load_test()
            end_time = datetime.now()
            
            results.update({
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "success": True
            })
            
            return results
            
        except Exception as e:
            end_time = datetime.now()
            return {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": str(e),
                "success": False
            }
    
    def _generate_scenario_recommendations(self, scenario: LoadTestScenario, 
                                         load_results: Dict[str, Any], 
                                         monitoring_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scenario results"""
        recommendations = []
        
        # Check if test was successful
        if not load_results.get("success", False):
            recommendations.append("❌ Load test failed to complete successfully. Check system stability and configuration.")
        
        # Check monitoring results
        if "system_metrics" in monitoring_results:
            system_metrics = monitoring_results["system_metrics"]
            
            if system_metrics.get("cpu", {}).get("avg", 0) > 80:
                recommendations.append("🔥 High CPU usage during test. Consider horizontal scaling or CPU optimization.")
            
            if system_metrics.get("memory", {}).get("avg", 0) > 85:
                recommendations.append("💾 High memory usage detected. Check for memory leaks or increase available memory.")
        
        # Check application metrics
        if "application_metrics" in monitoring_results:
            app_metrics = monitoring_results["application_metrics"]
            
            if app_metrics.get("overall_error_rate", 0) > 5:
                recommendations.append(f"⚠️ High error rate ({app_metrics['overall_error_rate']:.1f}%). Investigate error causes.")
        
        # Scenario-specific recommendations
        if scenario.name == "Stress Test":
            if load_results.get("success", False):
                recommendations.append("✅ System handled stress test well. Consider increasing load for capacity planning.")
            else:
                recommendations.append("🔴 System failed under stress. Identify bottlenecks and improve resilience.")
        
        elif scenario.name == "Spike Test":
            recommendations.append("📊 Review auto-scaling policies to handle traffic spikes better.")
        
        elif scenario.name == "Endurance Test":
            recommendations.append("🔄 Monitor for memory leaks and resource degradation over time.")
        
        if not recommendations:
            recommendations.append("✅ Test completed successfully with good performance metrics!")
        
        return recommendations
    
    def _save_results(self, scenario_name: str, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name.lower().replace(' ', '_')}_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("📊 LOAD TEST RESULTS SUMMARY")
        print("="*60)
        
        scenario = results["scenario"]
        print(f"🎯 Scenario: {scenario['name']}")
        print(f"⏱️  Duration: {scenario['duration_minutes']} minutes")
        
        # Load test results
        load_results = results.get("load_test_results", {})
        if load_results.get("success"):
            print("✅ Load Test: PASSED")
        else:
            print("❌ Load Test: FAILED")
            if "error" in load_results:
                print(f"   Error: {load_results['error']}")
        
        # Monitoring results summary
        monitoring = results.get("monitoring_results", {})
        if "system_metrics" in monitoring:
            sys_metrics = monitoring["system_metrics"]
            print(f"💻 Avg CPU: {sys_metrics.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"💾 Avg Memory: {sys_metrics.get('memory', {}).get('avg', 0):.1f}%")
        
        if "application_metrics" in monitoring:
            app_metrics = monitoring["application_metrics"]
            print(f"📈 Total Requests: {app_metrics.get('total_requests', 0)}")
            print(f"⚠️  Error Rate: {app_metrics.get('overall_error_rate', 0):.2f}%")
        
        # Alerts
        alerts = monitoring.get("alerts", [])
        if alerts:
            print(f"🚨 Alerts Triggered: {len(alerts)}")
            for alert in alerts[-3:]:  # Show last 3 alerts
                print(f"   - {alert.get('message', 'Unknown alert')}")
        else:
            print("🟢 No Alerts Triggered")
        
        # Recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("="*60)
    
    def run_test_suite(self, suite_name: str = "comprehensive"):
        """Run a predefined suite of load tests"""
        suites = {
            "quick": ["health_check", "daily_traffic"],
            "standard": ["health_check", "daily_traffic", "peak_usage", "database_focused"],
            "comprehensive": ["health_check", "daily_traffic", "peak_usage", "stress_test", "database_focused", "recommendation_pipeline"],
            "performance": ["daily_traffic", "peak_usage", "stress_test", "endurance_test"],
            "stability": ["endurance_test", "spike_test", "user_feedback_heavy"]
        }
        
        if suite_name not in suites:
            raise ValueError(f"Unknown test suite: {suite_name}. Available: {list(suites.keys())}")
        
        scenarios_to_run = suites[suite_name]
        
        print(f"\n🎯 Running Test Suite: {suite_name.upper()}")
        print(f"📋 Scenarios: {', '.join(scenarios_to_run)}")
        print(f"⏱️  Estimated Duration: {sum(self.scenarios[s].duration_minutes for s in scenarios_to_run)} minutes")
        
        suite_results = {}
        
        for i, scenario_name in enumerate(scenarios_to_run, 1):
            print(f"\n[{i}/{len(scenarios_to_run)}] Running scenario: {scenario_name}")
            
            try:
                result = self.run_scenario(scenario_name)
                suite_results[scenario_name] = result
                
                # Brief pause between tests
                if i < len(scenarios_to_run):
                    print("⏸️  Pausing 30 seconds between tests...")
                    time.sleep(30)
                    
            except Exception as e:
                print(f"❌ Failed to run scenario {scenario_name}: {e}")
                suite_results[scenario_name] = {"error": str(e), "success": False}
        
        # Save suite results
        suite_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_file = self.results_dir / f"test_suite_{suite_name}_{suite_timestamp}.json"
        
        with open(suite_file, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        print(f"\n🎯 Test Suite Complete! Results saved to: {suite_file}")
        
        return suite_results
    
    def list_scenarios(self):
        """List all available scenarios"""
        print("\n📋 Available Load Testing Scenarios:")
        print("-" * 50)
        
        for name, scenario in self.scenarios.items():
            print(f"🎯 {name}")
            print(f"   Description: {scenario.description}")
            print(f"   Duration: {scenario.duration_minutes} min | Max Users: {scenario.max_users}")
            print()

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="YouTube Recommendation System Load Testing")
    parser.add_argument("--scenario", help="Specific scenario to run")
    parser.add_argument("--suite", help="Test suite to run (quick, standard, comprehensive, performance, stability)")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--host", default="http://localhost:8080", help="Target host for load testing")
    parser.add_argument("--users", type=int, help="Override max users")
    parser.add_argument("--duration", type=int, help="Override duration in minutes")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner()
    
    if args.list:
        runner.list_scenarios()
        return
    
    # Custom parameters
    custom_params = {}
    if args.users:
        custom_params["users"] = args.users
    if args.duration:
        custom_params["run-time"] = f"{args.duration}m"
    
    try:
        if args.scenario:
            runner.run_scenario(args.scenario, custom_params)
        elif args.suite:
            runner.run_test_suite(args.suite)
        else:
            print("Please specify either --scenario or --suite. Use --list to see available options.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Load testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()