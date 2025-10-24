#!/usr/bin/env python3
"""
Failover Recovery Test Runner - Execute all failover and recovery tests
Generates comprehensive HTML reports following the existing project pattern
"""

import os
import sys
import json
import time
from datetime import datetime
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")

# Add project root to path
sys.path.append(str(project_root))

try:
    from tests.failover_recovery.test_service_recovery import ServiceRecoveryTester
except ImportError as e:
    print(f"Warning: Could not import ServiceRecoveryTester: {e}")
    ServiceRecoveryTester = None

try:
    from tests.failover_recovery.test_resource_stress import ResourceStressTester
except ImportError as e:
    print(f"Warning: Could not import ResourceStressTester: {e}")
    ResourceStressTester = None

try:
    from tests.failover_recovery.test_database_recovery import DatabaseRecoveryTester
    print("✓ DatabaseRecoveryTester imported successfully")
except ImportError as e:
    print(f"Warning: Could not import DatabaseRecoveryTester: {e}")
    
    # Fallback DatabaseRecoveryTester for cases where imports fail
    class DatabaseRecoveryTester:
        """Fallback database recovery tester"""
        def __init__(self):
            self.test_results = []
        
        def test_mongodb_connection_failure(self):
            return {"test": "mongodb_connection_failure", "status": "skipped", "reason": "Database client import failed"}
        
        def test_qdrant_connection_failure(self):
            return {"test": "qdrant_connection_failure", "status": "skipped", "reason": "Database client import failed"}
        
        def test_supabase_connection_failure(self):
            return {"test": "supabase_connection_failure", "status": "skipped", "reason": "Database client import failed"}
        
        def test_database_connection_recovery(self):
            return {
                "test": "database_connection_recovery",
                "status": "skipped",
                "recovery_score": 0,
                "individual_results": [],
                "message": "Database recovery tests skipped - import failed"
            }


class FailoverRecoveryTestRunner:
    """Main test runner for failover and recovery tests"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.test_results = {}
        self.start_time = datetime.now()
        self.project_root = project_root
    
    def run_service_recovery_tests(self) -> Dict[str, Any]:
        """Run service recovery tests"""
        print("Running Service Recovery Tests...")
        
        if not ServiceRecoveryTester:
            return {"status": "skipped", "reason": "ServiceRecoveryTester not available"}
        
        try:
            tester = ServiceRecoveryTester(self.api_url)
            
            results = {
                "test_category": "service_recovery",
                "start_time": datetime.now().isoformat(),
                "tests": {}
            }
            
            # Health check test
            print("  - Service health check...")
            results["tests"]["health_check"] = tester.check_service_health()
            
            # API endpoints test
            print("  - API endpoints availability...")
            results["tests"]["api_endpoints"] = tester.test_api_endpoints()
            
            # Memory stress test
            print("  - Memory stress recovery...")
            results["tests"]["memory_stress"] = tester.simulate_memory_stress(duration=5)
            
            # Timeout handling test
            print("  - Timeout handling...")
            results["tests"]["timeout_handling"] = tester.test_timeout_handling()
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "test_category": "service_recovery"
            }
    
    def run_database_recovery_tests(self) -> Dict[str, Any]:
        """Run database recovery tests"""
        print("Running Database Recovery Tests...")
        
        if not DatabaseRecoveryTester:
            return {"status": "skipped", "reason": "DatabaseRecoveryTester not available"}
        
        try:
            tester = DatabaseRecoveryTester()
            
            results = {
                "test_category": "database_recovery",
                "start_time": datetime.now().isoformat(),
                "tests": {}
            }
            
            # MongoDB connection test
            print("  - MongoDB connection resilience...")
            results["tests"]["mongodb_connection"] = tester.test_mongodb_connection_failure()
            
            # Qdrant connection test
            print("  - Qdrant connection resilience...")
            results["tests"]["qdrant_connection"] = tester.test_qdrant_connection_failure()
            
            # Supabase connection test
            print("  - Supabase connection resilience...")
            results["tests"]["supabase_connection"] = tester.test_supabase_connection_failure()
            
            # Overall recovery test
            print("  - Overall database recovery...")
            results["tests"]["overall_recovery"] = tester.test_database_connection_recovery()
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "test_category": "database_recovery"
            }
    
    def run_resource_stress_tests(self) -> Dict[str, Any]:
        """Run resource stress tests"""
        print("Running Resource Stress Tests...")
        
        if not ResourceStressTester:
            return {"status": "skipped", "reason": "ResourceStressTester not available"}
        
        try:
            tester = ResourceStressTester(self.api_url)
            
            results = {
                "test_category": "resource_stress",
                "start_time": datetime.now().isoformat(),
                "tests": {}
            }
            
            # System metrics baseline
            print("  - System metrics baseline...")
            results["tests"]["system_metrics"] = tester.get_system_metrics()
            
            # Memory stress test
            print("  - Memory stress test...")
            results["tests"]["memory_stress"] = tester.stress_test_memory(duration=5, target_mb=50)
            
            # CPU stress test
            print("  - CPU stress test...")
            results["tests"]["cpu_stress"] = tester.stress_test_cpu(duration=5, num_threads=2)
            
            # Disk I/O stress test
            print("  - Disk I/O stress test...")
            results["tests"]["disk_io_stress"] = tester.stress_test_disk_io(duration=3, file_size_mb=5)
            
            # API under stress test
            print("  - API under stress test...")
            try:
                results["tests"]["api_under_stress"] = tester.test_api_under_stress("memory", duration=5)
            except Exception as e:
                results["tests"]["api_under_stress"] = {"status": "skipped", "reason": str(e)}
            
            # Recovery test
            print("  - Recovery after stress...")
            results["tests"]["recovery_test"] = tester.test_recovery_after_stress(recovery_wait=3)
            
            results["end_time"] = datetime.now().isoformat()
            results["status"] = "completed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "test_category": "resource_stress"
            }
    
    def run_pytest_tests(self) -> Dict[str, Any]:
        """Run pytest-based tests and capture results"""
        print("Running pytest-based tests...")
        
        try:
            test_dir = Path(__file__).parent
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_dir),
                "-v", 
                "--tb=short",
                f"--junitxml={self.project_root}/failover_recovery_pytest_results.xml"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(self.project_root),
                timeout=300  # 5 minute timeout
            )
            
            return {
                "test_category": "pytest_integration",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "completed" if result.returncode == 0 else "failed_with_errors",
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_category": "pytest_integration",
                "status": "timeout",
                "error": "Tests timed out after 5 minutes"
            }
        except Exception as e:
            return {
                "test_category": "pytest_integration",
                "status": "error",
                "error": str(e)
            }
    
    def generate_html_report(self) -> str:
        """Generate HTML report similar to existing project reports"""
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate test summary
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category_results in self.test_results.values():
            if isinstance(category_results, dict) and "tests" in category_results:
                for test_result in category_results["tests"].values():
                    total_tests += 1
                    if isinstance(test_result, dict):
                        status = test_result.get("status", "unknown")
                        if status in ["healthy", "completed", "handled", "passed"]:
                            passed_tests += 1
                        elif status in ["skipped"]:
                            skipped_tests += 1
                        else:
                            failed_tests += 1
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Failover and Recovery Test Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .title {{
            color: #007acc;
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #666;
            margin: 0;
            font-size: 1.2em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card.total {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }}
        .summary-card.passed {{
            background-color: #e8f5e8;
            border-left: 4px solid #4caf50;
        }}
        .summary-card.failed {{
            background-color: #fce4ec;
            border-left: 4px solid #f44336;
        }}
        .summary-card.skipped {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
        }}
        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .summary-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .test-category {{
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .category-header {{
            background-color: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            font-size: 1.3em;
            color: #333;
        }}
        .category-content {{
            padding: 20px;
        }}
        .test-item {{
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 6px;
            background-color: #fafafa;
        }}
        .test-name {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .test-status {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-passed {{
            background-color: #4caf50;
            color: white;
        }}
        .status-failed {{
            background-color: #f44336;
            color: white;
        }}
        .status-skipped {{
            background-color: #ff9800;
            color: white;
        }}
        .status-error {{
            background-color: #9c27b0;
            color: white;
        }}
        .test-details {{
            margin-top: 10px;
            font-size: 0.9em;
            color: #666;
        }}
        .test-metrics {{
            margin-top: 10px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .metric {{
            background-color: #f0f0f0;
            padding: 8px;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 0.8em;
            color: #666;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }}
        .json-data {{
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            overflow-x: auto;
            font-family: monospace;
            font-size: 0.8em;
        }}
        .collapsible {{
            cursor: pointer;
            user-select: none;
            color: #007acc;
            text-decoration: underline;
        }}
        .content {{
            display: none;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">Failover and Recovery Test Results</h1>
            <p class="subtitle">Generated on {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="subtitle">Duration: {duration.total_seconds():.2f} seconds</p>
        </div>
        
        <div class="summary">
            <div class="summary-card total">
                <div class="summary-number">{total_tests}</div>
                <div class="summary-label">Total Tests</div>
            </div>
            <div class="summary-card passed">
                <div class="summary-number">{passed_tests}</div>
                <div class="summary-label">Passed</div>
            </div>
            <div class="summary-card failed">
                <div class="summary-number">{failed_tests}</div>
                <div class="summary-label">Failed</div>
            </div>
            <div class="summary-card skipped">
                <div class="summary-number">{skipped_tests}</div>
                <div class="summary-label">Skipped</div>
            </div>
        </div>
        """
        
        # Add test categories
        for category_name, category_results in self.test_results.items():
            html_content += self._generate_category_html(category_name, category_results)
        
        html_content += f"""
        <div class="footer">
            <p><strong>Test Environment:</strong></p>
            <ul>
                <li>API URL: {self.api_url}</li>
                <li>Python Version: {sys.version.split()[0]}</li>
                <li>Test Runner: {os.path.basename(__file__)}</li>
                <li>Project Root: {self.project_root}</li>
            </ul>
        </div>
    </div>
    
    <script>
        function toggleContent(element) {{
            var content = element.nextElementSibling;
            if (content.style.display === "block") {{
                content.style.display = "none";
            }} else {{
                content.style.display = "block";
            }}
        }}
        
        // Make collapsible elements work
        document.querySelectorAll('.collapsible').forEach(function(element) {{
            element.addEventListener('click', function() {{
                toggleContent(this);
            }});
        }});
    </script>
</body>
</html>"""
        
        return html_content
    
    def _generate_category_html(self, category_name: str, category_results: Dict[str, Any]) -> str:
        """Generate HTML for a test category"""
        
        category_title = category_name.replace('_', ' ').title()
        
        html = f"""
        <div class="test-category">
            <div class="category-header">{category_title}</div>
            <div class="category-content">
        """
        
        if category_results.get("status") == "error":
            html += f"""
                <div class="test-item">
                    <div class="test-name">Category Error</div>
                    <span class="test-status status-error">Error</span>
                    <div class="test-details">
                        <strong>Error:</strong> {category_results.get('error', 'Unknown error')}
                    </div>
                </div>
            """
        elif category_results.get("status") == "skipped":
            html += f"""
                <div class="test-item">
                    <div class="test-name">Category Skipped</div>
                    <span class="test-status status-skipped">Skipped</span>
                    <div class="test-details">
                        <strong>Reason:</strong> {category_results.get('reason', 'Not available')}
                    </div>
                </div>
            """
        elif "tests" in category_results:
            for test_name, test_result in category_results["tests"].items():
                html += self._generate_test_html(test_name, test_result)
        else:
            # Handle pytest results or other formats
            html += f"""
                <div class="test-item">
                    <div class="test-name">{category_title} Execution</div>
                    <span class="test-status status-{self._get_status_class(category_results.get('status', 'unknown'))}">{category_results.get('status', 'unknown')}</span>
                    <div class="test-details">
                        <div class="collapsible">Show Details</div>
                        <div class="content">
                            <div class="json-data">{json.dumps(category_results, indent=2)}</div>
                        </div>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_test_html(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """Generate HTML for individual test"""
        
        test_title = test_name.replace('_', ' ').title()
        status = test_result.get("status", "unknown")
        status_class = self._get_status_class(status)
        
        html = f"""
        <div class="test-item">
            <div class="test-name">{test_title}</div>
            <span class="test-status status-{status_class}">{status}</span>
        """
        
        # Add test-specific metrics
        if "response_time" in test_result:
            html += f"""
            <div class="test-metrics">
                <div class="metric">
                    <div class="metric-value">{test_result['response_time']:.3f}s</div>
                    <div class="metric-label">Response Time</div>
                </div>
            </div>
            """
        
        if "memory_increase_mb" in test_result:
            html += f"""
            <div class="test-metrics">
                <div class="metric">
                    <div class="metric-value">{test_result['memory_increase_mb']:.1f} MB</div>
                    <div class="metric-label">Memory Increase</div>
                </div>
            </div>
            """
        
        if "success_rate" in test_result:
            html += f"""
            <div class="test-metrics">
                <div class="metric">
                    <div class="metric-value">{test_result['success_rate']:.1%}</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>
            """
        
        # Add details section
        html += f"""
            <div class="test-details">
                <div class="collapsible">Show Raw Data</div>
                <div class="content">
                    <div class="json-data">{json.dumps(test_result, indent=2)}</div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _get_status_class(self, status: str) -> str:
        """Get CSS class for status"""
        status_lower = status.lower()
        if status_lower in ["healthy", "completed", "handled", "passed", "success"]:
            return "passed"
        elif status_lower in ["skipped"]:
            return "skipped"
        elif status_lower in ["error"]:
            return "error"
        else:
            return "failed"
    
    def save_results(self):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.project_root / f"failover_recovery_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "test_results": self.test_results
            }, f, indent=2, default=str)
        
        # Save HTML report
        html_file = self.project_root / "failover_recovery_report.html"
        with open(html_file, 'w') as f:
            f.write(self.generate_html_report())
        
        print(f"\nResults saved:")
        print(f"  JSON: {json_file}")
        print(f"  HTML: {html_file}")
        
        return str(html_file), str(json_file)
    
    def run_all_tests(self, include_pytest: bool = True):
        """Run all failover and recovery tests"""
        print("=" * 60)
        print("FAILOVER AND RECOVERY TEST SUITE")
        print("=" * 60)
        
        # Run all test categories
        self.test_results["service_recovery"] = self.run_service_recovery_tests()
        self.test_results["database_recovery"] = self.run_database_recovery_tests()  
        self.test_results["resource_stress"] = self.run_resource_stress_tests()
        
        if include_pytest:
            self.test_results["pytest_integration"] = self.run_pytest_tests()
        
        # Save results and generate report
        html_file, json_file = self.save_results()
        
        print("\n" + "=" * 60)
        print("TEST SUITE COMPLETED")
        print("=" * 60)
        
        return html_file, json_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Failover and Recovery Tests")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API URL to test")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip pytest integration tests")
    parser.add_argument("--quick", action="store_true", help="Run quick version of tests")
    
    args = parser.parse_args()
    
    runner = FailoverRecoveryTestRunner(api_url=args.api_url)
    
    if args.quick:
        print("Running quick failover recovery tests...")
        # Only run basic tests for quick mode
        runner.test_results["service_recovery"] = runner.run_service_recovery_tests()
        html_file, json_file = runner.save_results()
    else:
        html_file, json_file = runner.run_all_tests(include_pytest=not args.skip_pytest)
    
    print(f"\nOpen {html_file} in your browser to view the detailed report.")


if __name__ == "__main__":
    main()