"""
Security Test Runner for Background Service

Utilities for running security tests focused on background/microservice security concerns
rather than multi-user authentication/authorization systems.
"""

import pytest
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests

# Add the parent directories to Python path to handle imports
script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(tests_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, tests_dir)

try:
    from tests.security.security_test_config import SecurityTestConfig
except ImportError:
    from security_test_config import SecurityTestConfig


class BackgroundServiceSecurityRunner:
    """Security test runner for background services"""
    
    def __init__(self):
        self.config = SecurityTestConfig()
        self.results = {
            "test_run_info": {
                "timestamp": datetime.now().isoformat(),
                "target_system": self.config.BASE_URL,
                "service_type": "background_microservice",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "issues_found": []
            },
            "test_categories": {
                "service_availability": {"status": "not_run", "issues": []},
                "input_validation": {"status": "not_run", "issues": []}, 
                "resource_management": {"status": "not_run", "issues": []},
                "operational_security": {"status": "not_run", "issues": []},
                "dos_protection": {"status": "not_run", "issues": []}
            }
        }
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run security scan for background service"""
        print("\\n🔒 Background Service Security Scan")
        print("=" * 50)
        
        # Check if server is running
        if not self.is_server_available():
            print("❌ Service is not available. Please start the service first.")
            return self.results
        
        # Run relevant tests for background services
        self.run_background_service_assessment()
        
        return self.results
    
    def is_server_available(self) -> bool:
        """Check if the target service is available"""
        try:
            response = requests.get(f"{self.config.BASE_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_background_service_assessment(self):
        """Run assessment focused on background service security"""
        print("\\n�️ Background Service Security Assessment")
        print("-" * 45)
        
        # Test 1: Service availability and health
        print("\\n1. Testing Service Availability...")
        availability_issues = self.test_service_availability()
        if availability_issues:
            self.results["test_run_info"]["issues_found"].extend(availability_issues)
            self.results["test_categories"]["service_availability"]["issues"] = availability_issues
            self.results["test_categories"]["service_availability"]["status"] = "issues_found"
        else:
            self.results["test_categories"]["service_availability"]["status"] = "healthy"
        
        # Test 2: Input validation and service stability
        print("\\n2. Testing Input Validation...")
        input_issues = self.test_input_validation()
        if input_issues:
            self.results["test_run_info"]["issues_found"].extend(input_issues)
            self.results["test_categories"]["input_validation"]["issues"] = input_issues
            self.results["test_categories"]["input_validation"]["status"] = "vulnerable"
        else:
            self.results["test_categories"]["input_validation"]["status"] = "robust"
        
        # Test 3: Resource management
        print("\\n3. Testing Resource Management...")
        resource_issues = self.test_resource_management()
        if resource_issues:
            self.results["test_run_info"]["issues_found"].extend(resource_issues)
            self.results["test_categories"]["resource_management"]["issues"] = resource_issues
            self.results["test_categories"]["resource_management"]["status"] = "vulnerable"
        else:
            self.results["test_categories"]["resource_management"]["status"] = "protected"
        
        # Test 4: DoS protection
        print("\\n4. Testing DoS Protection...")
        dos_issues = self.test_dos_protection()
        if dos_issues:
            self.results["test_run_info"]["issues_found"].extend(dos_issues)
            self.results["test_categories"]["dos_protection"]["issues"] = dos_issues
            self.results["test_categories"]["dos_protection"]["status"] = "vulnerable"
        else:
            self.results["test_categories"]["dos_protection"]["status"] = "protected"
        
        # Generate summary
        self.generate_security_summary()
    
    def test_service_availability(self) -> List[Dict[str, Any]]:
        """Test service health and availability"""
        issues = []
        
        health_endpoints = ["/health", "/newsletter/health"]
        
        for endpoint in health_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.config.BASE_URL}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code != 200:
                    issues.append({
                        "severity": "MEDIUM",
                        "type": "Health Check Issue",
                        "endpoint": endpoint,
                        "description": f"Health endpoint {endpoint} returned {response.status_code}",
                        "impact": "Service health monitoring may be compromised"
                    })
                elif response_time > 5.0:
                    issues.append({
                        "severity": "LOW",
                        "type": "Slow Health Check",
                        "endpoint": endpoint,
                        "description": f"Health check took {response_time:.2f}s (too slow)",
                        "impact": "Health monitoring may not detect issues quickly"
                    })
            except:
                issues.append({
                    "severity": "HIGH",
                    "type": "Health Check Failure",
                    "endpoint": endpoint,
                    "description": f"Health endpoint {endpoint} is unreachable",
                    "impact": "Cannot monitor service health"
                })
        
        print(f"   Found {len(issues)} availability issues")
        return issues
    
    def test_input_validation(self) -> List[Dict[str, Any]]:
        """Test input validation for service stability"""
        issues = []
        
        # Test malformed inputs that could crash service
        test_cases = [
            ("", "Empty user_id"),
            ("x" * 1000, "Very long user_id"),
            ("../../../etc/passwd", "Path traversal attempt"),
            ("<script>alert('xss')</script>", "XSS payload")
        ]
        
        for payload, description in test_cases:
            try:
                response = requests.get(
                    f"{self.config.BASE_URL}/recommendations/?user_id={payload}&top_k=5",
                    timeout=10
                )
                
                if response.status_code == 500:
                    issues.append({
                        "severity": "MEDIUM",
                        "type": "Input Validation Issue",
                        "endpoint": "/recommendations/",
                        "description": f"Service crashes with {description}",
                        "impact": "Service instability, potential DoS"
                    })
            except:
                continue
        
        print(f"   Found {len(issues)} input validation issues")
        return issues
    
    def test_resource_management(self) -> List[Dict[str, Any]]:
        """Test resource management and limits"""
        issues = []
        
        # Test large payload handling
        try:
            large_payload = {"user_id": "test", "large_data": "x" * 100000}  # 100KB
            response = requests.post(
                f"{self.config.BASE_URL}/run-workflow/run-workflow",
                json=large_payload,
                timeout=20
            )
            
            if response.status_code not in [200, 400, 413]:  # Should handle gracefully
                issues.append({
                    "severity": "MEDIUM",
                    "type": "Large Payload Handling",
                    "endpoint": "/run-workflow/run-workflow",
                    "description": f"Service doesn't handle large payloads gracefully: {response.status_code}",
                    "impact": "Memory exhaustion, service crashes"
                })
        except requests.exceptions.Timeout:
            issues.append({
                "severity": "HIGH",
                "type": "Resource Timeout",
                "endpoint": "/run-workflow/run-workflow",
                "description": "Service times out with large payloads",
                "impact": "Resource exhaustion, service unavailability"
            })
        except:
            pass
        
        print(f"   Found {len(issues)} resource management issues")
        return issues
    
    def test_dos_protection(self) -> List[Dict[str, Any]]:
        """Test DoS protection mechanisms"""
        issues = []
        
        # Test rapid requests
        rapid_success = 0
        for i in range(10):
            try:
                response = requests.get(
                    f"{self.config.BASE_URL}/recommendations/?user_id=dos_test&top_k=5",
                    timeout=2
                )
                if response.status_code == 200:
                    rapid_success += 1
                time.sleep(0.1)
            except:
                break
        
        if rapid_success >= 8:  # Most requests succeeded
            issues.append({
                "severity": "MEDIUM",
                "type": "No Rate Limiting",
                "endpoint": "/recommendations/",
                "description": f"No rate limiting detected - {rapid_success}/10 rapid requests succeeded",
                "impact": "Service vulnerable to DoS attacks"
            })
        
        print(f"   Found {len(issues)} DoS protection issues")
        return issues
    
    def generate_security_summary(self):
        """Generate and display security summary"""
        print("\\n" + "=" * 50)
        print("🛡️ BACKGROUND SERVICE SECURITY SUMMARY")
        print("=" * 50)
        
        total_issues = len(self.results["test_run_info"]["issues_found"])
        
        # Count by severity
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in self.results["test_run_info"]["issues_found"]:
            severity_counts[issue["severity"]] += 1
        
        print(f"\\n📊 SECURITY ASSESSMENT RESULTS:")
        print(f"   Total Issues Found: {total_issues}")
        for severity, count in severity_counts.items():
            if count > 0:
                icon = "🚨" if severity == "HIGH" else "⚠️" if severity == "MEDIUM" else "ℹ️"
                print(f"   {icon} {severity}: {count}")
        
        if total_issues == 0:
            print("   ✅ No significant security issues found!")
        
        # Show recommendations based on background service needs
        print(f"\\n🔧 BACKGROUND SERVICE RECOMMENDATIONS:")
        if any(issue["severity"] in ["HIGH", "MEDIUM"] for issue in self.results["test_run_info"]["issues_found"]):
            print("   1. 🛡️  Implement input validation and sanitization")
            print("   2. ⚡ Add request rate limiting")
            print("   3. 📊 Improve health monitoring")
            print("   4. 💾 Add payload size limits")
            print("   5. 📝 Implement structured error handling")
        else:
            print("   ✅ Service appears to be well-configured for security")
            print("   📊 Continue monitoring service health")
            print("   📝 Keep error handling informative but not verbose")
        
        print(f"\\n📋 SERVICE-SPECIFIC NOTES:")
        print("   • This is a background service - no user authentication needed")
        print("   • Focus on service stability and resource management")
        print("   • Monitor for DoS and resource exhaustion attacks")
        print("   • Ensure graceful degradation when dependencies fail")


def run_background_service_security_scan():
    """Convenience function to run background service security scan"""
    runner = BackgroundServiceSecurityRunner()
    return runner.run_security_scan()


if __name__ == "__main__":
    # Run security scan when script is executed directly
    run_background_service_security_scan()
    
    def test_authentication_bypass(self) -> List[Dict[str, Any]]:
        """Test for authentication bypass vulnerabilities"""
        issues = []
        
        # Test endpoints without authentication
        test_endpoints = [
            "/recommendations/?user_id=test&top_k=5",
            "/user-vector-update/status",
            "/test/debug-database"
        ]
        
        for endpoint in test_endpoints:
            try:
                response = requests.get(f"{self.config.BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    issues.append({
                        "severity": "HIGH",
                        "type": "Authentication Bypass",
                        "endpoint": endpoint,
                        "description": f"Endpoint {endpoint} accessible without authentication",
                        "impact": "Unauthorized access to system functionality"
                    })
            except:
                continue
        
        print(f"   Found {len(issues)} authentication bypass issues")
        return issues
    
    def test_authorization_flaws(self) -> List[Dict[str, Any]]:
        """Test for authorization vulnerabilities"""
        issues = []
        
        # Test horizontal privilege escalation
        test_users = ["user1", "user2", "admin", "guest"]
        
        for user in test_users:
            try:
                response = requests.get(
                    f"{self.config.BASE_URL}/recommendations/?user_id={user}&top_k=5",
                    timeout=5
                )
                if response.status_code == 200:
                    issues.append({
                        "severity": "CRITICAL",
                        "type": "Horizontal Privilege Escalation", 
                        "endpoint": "/recommendations/",
                        "description": f"Can access user '{user}' data without authorization",
                        "impact": "Access to other users' private data"
                    })
            except:
                continue
        
        print(f"   Found {len(issues)} authorization issues")
        return issues
    
    def test_input_validation_flaws(self) -> List[Dict[str, Any]]:
        """Test for input validation vulnerabilities"""
        issues = []
        
        # Test SQL injection
        sql_payloads = ["'; DROP TABLE users; --", "' OR '1'='1' --"]
        
        for payload in sql_payloads:
            try:
                response = requests.get(
                    f"{self.config.BASE_URL}/recommendations/?user_id={payload}&top_k=5",
                    timeout=10
                )
                
                if response.status_code == 500 or 'error' in response.text.lower():
                    issues.append({
                        "severity": "HIGH",
                        "type": "SQL Injection",
                        "endpoint": "/recommendations/",
                        "description": f"Potential SQL injection with payload: {payload[:20]}...",
                        "impact": "Database compromise, data theft"
                    })
            except:
                continue
        
        # Test XSS
        xss_payload = "<script>alert('XSS')</script>"
        try:
            response = requests.get(
                f"{self.config.BASE_URL}/recommendations/?user_id={xss_payload}&top_k=5",
                timeout=5
            )
            if xss_payload in response.text:
                issues.append({
                    "severity": "MEDIUM",
                    "type": "Reflected XSS",
                    "endpoint": "/recommendations/",
                    "description": "XSS payload reflected in response",
                    "impact": "Client-side code execution, session hijacking"
                })
        except:
            pass
        
        print(f"   Found {len(issues)} input validation issues")
        return issues
    
    def test_data_exposure(self) -> List[Dict[str, Any]]:
        """Test for data exposure vulnerabilities"""
        issues = []
        
        # Test error message exposure
        try:
            response = requests.get(f"{self.config.BASE_URL}/recommendations/?user_id=&top_k=abc", timeout=5)
            
            sensitive_patterns = ['password', 'secret', 'key', 'token', 'database', 'mongodb://']
            for pattern in sensitive_patterns:
                if pattern in response.text.lower():
                    issues.append({
                        "severity": "MEDIUM",
                        "type": "Information Disclosure",
                        "endpoint": "/recommendations/",
                        "description": f"Error message contains sensitive information: {pattern}",
                        "impact": "Information leakage, system fingerprinting"
                    })
                    break
        except:
            pass
        
        print(f"   Found {len(issues)} data exposure issues")
        return issues
    
    def test_rate_limiting(self) -> List[Dict[str, Any]]:
        """Test for rate limiting vulnerabilities"""
        issues = []
        
        # Test rapid requests
        rapid_requests = 0
        for i in range(10):
            try:
                response = requests.get(
                    f"{self.config.BASE_URL}/recommendations/?user_id=rate_test&top_k=5",
                    timeout=2
                )
                if response.status_code == 200:
                    rapid_requests += 1
                elif response.status_code == 429:
                    break
                time.sleep(0.1)
            except:
                break
        
        if rapid_requests >= 8:  # If most requests succeeded
            issues.append({
                "severity": "MEDIUM", 
                "type": "No Rate Limiting",
                "endpoint": "/recommendations/",
                "description": f"No rate limiting detected - {rapid_requests}/10 rapid requests succeeded",
                "impact": "Denial of service, resource exhaustion"
            })
        
        print(f"   Found {len(issues)} rate limiting issues")
        return issues
    
    def generate_security_summary(self):
        """Generate and display security summary"""
        print("\\n" + "=" * 60)
        print("🔒 SECURITY SCAN SUMMARY")
        print("=" * 60)
        
        total_issues = len(self.results["vulnerabilities_found"])
        
        # Count by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in self.results["vulnerabilities_found"]:
            severity_counts[issue["severity"]] += 1
        
        print(f"\\n📊 VULNERABILITY SUMMARY:")
        print(f"   Total Issues Found: {total_issues}")
        for severity, count in severity_counts.items():
            if count > 0:
                icon = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "HIGH" else "⚡" if severity == "MEDIUM" else "ℹ️"
                print(f"   {icon} {severity}: {count}")
        
        print(f"\\n🎯 TOP SECURITY ISSUES:")
        
        # Show top issues by severity
        critical_issues = [i for i in self.results["test_run_info"]["issues_found"] if i["severity"] == "HIGH"]  # Using HIGH as highest priority
        medium_issues = [i for i in self.results["test_run_info"]["issues_found"] if i["severity"] == "MEDIUM"]
        
        shown_issues = (critical_issues + medium_issues)[:5]  # Show top 5
        
        for i, issue in enumerate(shown_issues, 1):
            print(f"\\n   {i}. {issue['type']} [{issue['severity']}]")
            print(f"      Endpoint: {issue['endpoint']}")
            print(f"      Issue: {issue['description']}")
            print(f"      Impact: {issue['impact']}")
        
        if total_issues > 5:
            print(f"\\n   ... and {total_issues - 5} more issues")
        
        # Security recommendations
        print(f"\\n🔧 IMMEDIATE RECOMMENDATIONS:")
        if critical_issues or medium_issues:
            print("   1. 🚨 URGENT: Implement authentication and authorization")
            print("   2. 🛡️  Add input validation and sanitization") 
            print("   3. 🔒 Implement proper access controls")
            print("   4. 📝 Add request logging and monitoring")
            print("   5. ⚡ Implement rate limiting")
        else:
            print("   ✅ No critical issues found, but continue monitoring")
        
        print(f"\\n📋 DETAILED REPORT:")
        print(f"   A detailed security report can be found in the test results")
        print(f"   Run individual test modules for more specific analysis")


def run_security_scan():
    """Convenience function to run security scan"""
    runner = BackgroundServiceSecurityRunner()
    return runner.run_security_scan()


def generate_security_report(output_file: Optional[str] = None):
    """Generate a detailed security report"""
    runner = BackgroundServiceSecurityRunner()
    results = runner.run_security_scan()
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\\n📄 Security report saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run security scan when script is executed directly
    run_security_scan()